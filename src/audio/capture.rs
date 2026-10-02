//! Audio capture using the `cpal` crate.
//!
//! Opens the configured input device (see [`super::device`]) at 16kHz mono
//! 16-bit and pushes audio chunks into a tokio mpsc channel for downstream
//! processing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::device::{self, InputSource, ResolvedInput};
use super::{block_off_runtime, AudioChunk};

/// Desired sample rate for speech recognition.
pub const SAMPLE_RATE: u32 = 16_000;

/// Number of channels (mono).
pub(super) const CHANNELS: u16 = 1;

/// How long the caller waits for the capture thread to report that the
/// stream is running. Longer than [`device::PULSE_BUILD_TIMEOUT`], so a
/// stream-creation timeout surfaces as its own error first. It also covers
/// the cases cpal does not bound: a PulseAudio `play()` blocks until the
/// first data arrives, and listing sources on a hung server never returns.
const START_TIMEOUT: Duration = Duration::from_secs(5);

/// A handle to a running audio capture session.
///
/// The `cpal::Stream` lives on a dedicated capture thread.
/// This handle provides only the receiver and a stop signal.
pub struct AudioCaptureHandle {
    /// Receiver end of the audio channel.
    receiver: Option<mpsc::UnboundedReceiver<AudioChunk>>,
    /// Signal to stop the capture thread.
    stop_signal: Arc<AtomicBool>,
    /// Join handle for the capture thread.
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for AudioCaptureHandle {
    fn drop(&mut self) {
        // Signal the capture thread to stop.
        self.stop_signal.store(true, Ordering::Release);
        // Wait for the thread to finish (non-async, best-effort).
        if let Some(handle) = self.thread_handle.take() {
            handle.join().ok();
        }
    }
}

impl AudioCaptureHandle {
    /// Start capturing from `device` (the `[audio] device` value) and
    /// optionally publish a normalized volume level.
    ///
    /// The capture runs on a dedicated thread. Audio chunks are sent through
    /// the internal channel; call `take_receiver()` to get the receiving end.
    pub fn start_with_level_tx(
        device: &str,
        level_tx: Option<tokio::sync::watch::Sender<f32>>,
    ) -> anyhow::Result<Self> {
        let device_name = device.to_string();
        let device = device_name.clone();
        let (tx, rx) = mpsc::unbounded_channel::<AudioChunk>();
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_signal);

        // Channel to send back any initialization error from the thread.
        let (init_tx, init_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

        let thread_handle = std::thread::Builder::new()
            .name("whisrs-audio".into())
            .spawn(move || {
                run_capture(&device, tx, stop_clone, init_tx, level_tx);
            })
            .context("failed to spawn audio capture thread")?;

        // Wait for initialization result. Opening a device can take seconds
        // (a PulseAudio handshake times out at 2 s), so keep it off the
        // async workers, and never wait unbounded: callers hold the daemon
        // state lock across this.
        let init_result = match block_off_runtime(|| init_rx.recv_timeout(START_TIMEOUT)) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // The thread may be stuck inside cpal. Tell it to stop and
                // detach it (dropping the JoinHandle): it exits on its own
                // once cpal returns, and nothing here ever joins it.
                stop_signal.store(true, Ordering::Release);
                drop(thread_handle);
                return Err(start_timeout_error(&device_name));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("audio capture thread exited unexpectedly")
            }
        };
        init_result?;

        Ok(Self {
            receiver: Some(rx),
            stop_signal,
            thread_handle: Some(thread_handle),
        })
    }

    /// Take the receiver end of the audio channel.
    pub fn take_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<AudioChunk>> {
        self.receiver.take()
    }

    /// Signal the capture thread to stop (async-friendly).
    /// The channel will close once the thread exits. Callers reading
    /// from the receiver will see `None` after remaining chunks drain.
    pub fn stop(&mut self) {
        self.stop_signal.store(true, Ordering::Release);
    }

    /// Stop the audio capture and return all accumulated samples from the channel.
    pub async fn stop_and_collect(mut self) -> anyhow::Result<Vec<i16>> {
        // Signal the capture thread to stop.
        self.stop_signal.store(true, Ordering::Release);

        // Wait for the thread to finish.
        if let Some(handle) = self.thread_handle.take() {
            // Use spawn_blocking to avoid blocking the tokio runtime.
            tokio::task::spawn_blocking(move || {
                handle.join().ok();
            })
            .await?;
        }

        let mut all_samples = Vec::new();

        if let Some(mut rx) = self.receiver.take() {
            // Drain all remaining chunks from the channel.
            rx.close();
            while let Ok(chunk) = rx.try_recv() {
                all_samples.extend_from_slice(&chunk);
            }
        }

        info!("captured {} audio samples", all_samples.len());
        Ok(all_samples)
    }
}

/// The error for a device that did not start within [`START_TIMEOUT`].
fn start_timeout_error(device: &str) -> anyhow::Error {
    let secs = START_TIMEOUT.as_secs();
    if device::is_default_name(device) {
        anyhow::anyhow!(
            "the default audio input device did not start within {secs}s; \
             check that the microphone is connected and the sound server is running"
        )
    } else {
        anyhow::anyhow!(
            "audio device {} did not start within {secs}s; \
             set [audio] device = \"default\"",
            device.trim()
        )
    }
}

/// How long after `play()` to wait for the first callback before warning.
const FIRST_CHUNK_TIMEOUT: Duration = Duration::from_secs(1);

/// Upper bound on [`drain_after_stop`]: two of PipeWire-ALSA's 128 ms
/// default periods (the `BufferSize::Default` fallback), so a stream that
/// stopped delivering cannot hold up the stop.
const STOP_DRAIN_TIMEOUT: Duration = Duration::from_millis(250);

/// After the stop signal, keep the stream until one more chunk arrives.
/// Audio reaches the callback a whole period at a time, so dropping the
/// stream right away loses the partial period in the device buffer: the
/// end of the last word. The next chunk covers the moment stop was seen.
fn drain_after_stop(chunks: &AtomicU64) {
    let seen = chunks.load(Ordering::Acquire);
    let started = Instant::now();
    while chunks.load(Ordering::Acquire) == seen && started.elapsed() < STOP_DRAIN_TIMEOUT {
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Run the audio capture on the current thread.
///
/// Sends the initialization result through `init_tx`, then blocks until the
/// stop signal is set. The cpal Stream lives on this thread.
fn run_capture(
    device: &str,
    tx: mpsc::UnboundedSender<AudioChunk>,
    stop_signal: Arc<AtomicBool>,
    init_tx: std::sync::mpsc::Sender<anyhow::Result<()>>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
) {
    let result = setup_and_run(device, tx, stop_signal, &init_tx, level_tx);
    if let Err(e) = result {
        // If init_tx hasn't been used yet, send the error.
        init_tx.send(Err(e)).ok();
    }
}

fn setup_and_run(
    device_name: &str,
    tx: mpsc::UnboundedSender<AudioChunk>,
    stop_signal: Arc<AtomicBool>,
    init_tx: &std::sync::mpsc::Sender<anyhow::Result<()>>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
) -> anyhow::Result<()> {
    // Non-empty chunks delivered so far.
    let chunks = Arc::new(AtomicU64::new(0));
    let resolved = device::resolve_input(device_name)?;
    let cached_pulse_host = resolved.cached_pulse_host;
    let (stream, device, source) = match build_and_play(resolved, &tx, &level_tx, &chunks) {
        Ok(opened) => opened,
        Err(e) if cached_pulse_host && device::is_host_level_error(&e) => {
            // The cached PulseAudio connection is stale (the server
            // restarted under it): reconnect once. A device-level error
            // (busy, unsupported config, timeout) keeps the connection.
            debug!(
                "opening input on the cached PulseAudio connection failed ({e:#}), reconnecting"
            );
            device::invalidate_pulse_host();
            let resolved = device::resolve_input(device_name)?;
            build_and_play(resolved, &tx, &level_tx, &chunks)?
        }
        Err(e) => return Err(e),
    };
    debug!("audio capture started at {SAMPLE_RATE}Hz mono i16");

    // Signal successful initialization. If the caller already gave up
    // (START_TIMEOUT), it set the stop signal and dropped the receiver:
    // release the stream and exit.
    if stop_signal.load(Ordering::Acquire) || init_tx.send(Ok(())).is_err() {
        debug!("audio capture started after the caller gave up; stopping");
        drop(stream);
        return Ok(());
    }

    // Block until stop is signaled. Keep the stream alive.
    let started = Instant::now();
    let mut watchdog_armed = true;
    while !stop_signal.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_millis(50));
        if watchdog_armed && started.elapsed() >= FIRST_CHUNK_TIMEOUT {
            watchdog_armed = false;
            if chunks.load(Ordering::Acquire) == 0 {
                warn_no_audio(&device, &source);
            }
        }
    }

    debug!("audio capture stopping");
    drain_after_stop(&chunks);
    if let Some(level_tx) = &level_tx {
        let _ = level_tx.send(0.0);
    }
    drop(stream);

    Ok(())
}

/// A stream that started but has delivered nothing. A stalled PulseAudio
/// source never gets here (its `play()` blocks until the first data, which
/// [`START_TIMEOUT`] reports), so this is about the device itself.
fn warn_no_audio(device: &str, source: &InputSource) {
    let secs = FIRST_CHUNK_TIMEOUT.as_secs();
    match source {
        InputSource::Named(..) => warn!(
            "no audio from input device {device} ({source}) {secs} s after starting; \
             check that the device is connected and not muted, or set \
             [audio] device = \"default\""
        ),
        _ => warn!(
            "no audio from input device {device} ({source}) {secs} s after starting; \
             check that the microphone is connected and not muted"
        ),
    }
}

/// Build and start the input stream for a resolved device.
fn build_and_play(
    resolved: ResolvedInput,
    tx: &mpsc::UnboundedSender<AudioChunk>,
    level_tx: &Option<tokio::sync::watch::Sender<f32>>,
    chunks: &Arc<AtomicU64>,
) -> anyhow::Result<(cpal::Stream, String, InputSource)> {
    let device = resolved.device;
    let device_label = device.to_string();
    info!(
        "using audio input device: {device_label} ({})",
        resolved.source
    );

    // Verify device support.
    let supported = device
        .supported_input_configs()
        .context("failed to query supported input configs")?;

    let mut found_match = false;
    for range in supported {
        if range.channels() == CHANNELS
            && range.min_sample_rate() <= SAMPLE_RATE
            && range.max_sample_rate() >= SAMPLE_RATE
            && range.sample_format() == SampleFormat::I16
        {
            found_match = true;
            break;
        }
    }

    if !found_match {
        warn!(
            "device may not natively support {SAMPLE_RATE}Hz mono i16; \
             cpal will attempt conversion"
        );
    }

    let build = |buffer_size: BufferSize| {
        let config = StreamConfig {
            channels: CHANNELS,
            sample_rate: SAMPLE_RATE,
            buffer_size,
        };
        let tx = tx.clone();
        let callback_level_tx = level_tx.clone();
        let chunks = Arc::clone(chunks);
        device.build_input_stream(
            config,
            move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                if let Some(level_tx) = &callback_level_tx {
                    let _ = level_tx.send(audio_level(data));
                }
                if tx.send(data.to_vec()).is_err() {
                    // Channel closed — capture is stopping.
                }
                // Counted after the send, so a drain that sees the chunk
                // knows it is already queued for the collector.
                if !data.is_empty() {
                    chunks.fetch_add(1, Ordering::Release);
                }
            },
            |err: cpal::Error| {
                error!("audio stream error: {err}");
            },
            resolved.build_timeout,
        )
    };

    let build_started = Instant::now();
    let stream = match build(resolved.buffer_size) {
        Err(e) if retry_with_default_buffer(&resolved.source, resolved.buffer_size) => {
            debug!(
                "opening the default input with {:?} failed ({e}), retrying with the device's default buffer",
                resolved.buffer_size
            );
            build(BufferSize::Default)
        }
        result => result,
    };
    let stream = stream.map_err(|e| match resolved.build_timeout {
        // cpal gave up waiting on the server (a hung PulseAudio).
        Some(limit) if build_started.elapsed() >= limit => anyhow::Error::new(e).context(format!(
            "audio device {} did not start within {}s; set [audio] device = \"default\"",
            match &resolved.source {
                InputSource::Named(_, id) => id.as_str(),
                _ => device_label.as_str(),
            },
            limit.as_secs()
        )),
        _ => anyhow::Error::new(e).context("failed to build audio input stream"),
    })?;

    stream.play().context("failed to start audio stream")?;
    Ok((stream, device_label, resolved.source))
}

/// Whether a failed open of an ALSA input should be retried with
/// `BufferSize::Default`: the fixed period ([`device::ALSA_DEFAULT_INPUT_FRAMES`])
/// is a latency choice, never a reason for an ALSA device to fail.
fn retry_with_default_buffer(source: &InputSource, buffer_size: BufferSize) -> bool {
    matches!(buffer_size, BufferSize::Fixed(_))
        && matches!(
            source,
            InputSource::Default
                | InputSource::Fallback
                | InputSource::Named(device::DeviceHost::Alsa, _)
        )
}

fn audio_level(data: &[i16]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = data
        .iter()
        .map(|sample| {
            let normalized = *sample as f32 / i16::MAX as f32;
            normalized * normalized
        })
        .sum();
    let rms = (sum_squares / data.len() as f32).sqrt();

    // Soft compressor: 1 - exp(-k*rms). k=18 maps typical speech RMS
    // (~0.05–0.15) to the 0.6–0.95 range, so the visualizer reaches the
    // top of its dynamic range during normal speech instead of hovering
    // around 30 % deflection.
    (1.0 - (-rms * 18.0).exp()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use device::DeviceHost;

    #[test]
    fn default_input_retries_with_default_buffer() {
        let fixed = BufferSize::Fixed(device::ALSA_DEFAULT_INPUT_FRAMES);
        assert!(retry_with_default_buffer(&InputSource::Default, fixed));
        assert!(retry_with_default_buffer(&InputSource::Fallback, fixed));
        // Already on the default buffer: nothing to retry.
        assert!(!retry_with_default_buffer(
            &InputSource::Default,
            BufferSize::Default
        ));
        // A named ALSA device also falls back; PulseAudio keeps its own error.
        assert!(retry_with_default_buffer(
            &InputSource::Named(DeviceHost::Alsa, "x".into()),
            fixed
        ));
        assert!(!retry_with_default_buffer(
            &InputSource::Named(DeviceHost::PulseAudio, "x".into()),
            fixed
        ));
    }

    #[test]
    fn drain_returns_on_next_chunk() {
        let chunks = Arc::new(AtomicU64::new(3));
        let feeder = Arc::clone(&chunks);
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            feeder.fetch_add(1, Ordering::Release);
        });
        let started = Instant::now();
        drain_after_stop(&chunks);
        t.join().unwrap();
        assert!(started.elapsed() < STOP_DRAIN_TIMEOUT);
        assert_eq!(chunks.load(Ordering::Acquire), 4);
    }

    #[test]
    fn drain_is_bounded_without_audio() {
        let started = Instant::now();
        drain_after_stop(&AtomicU64::new(0));
        assert!(started.elapsed() >= STOP_DRAIN_TIMEOUT);
        assert!(started.elapsed() < STOP_DRAIN_TIMEOUT + Duration::from_millis(200));
    }
}
