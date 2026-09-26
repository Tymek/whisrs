//! Interruptible WAV playback on the default audio output device.
//!
//! Used by the read-selection-aloud feature to play TTS audio of arbitrary
//! sample rate / channel count. Unlike [`crate::audio::feedback`] (mono /
//! 44.1 kHz / 2-second timeout), this plays the clip to completion and can be
//! stopped early via a shared [`AtomicBool`].

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};
use tracing::{debug, warn};

use crate::WhisrsError;

/// Decoded PCM audio: interleaved f32 samples plus the stream format.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedWav {
    /// Interleaved samples in `[-1.0, 1.0]`, channel-major per frame.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

impl DecodedWav {
    /// Number of frames (samples per channel).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }
}

/// Decode WAV bytes into interleaved f32 samples plus format metadata.
///
/// Handles 16-bit integer and 32-bit float WAV files (the formats Groq's TTS
/// endpoint returns); other integer bit depths (8/24/32) are also supported by
/// scaling to f32. This is a pure function with no audio device access so it
/// can be unit-tested.
pub fn decode_wav(wav_bytes: &[u8]) -> Result<DecodedWav, WhisrsError> {
    // Some encoders (ffmpeg/Lavf, which Groq's TTS endpoint uses) stream WAV to
    // a non-seekable output and leave the RIFF and `data` chunk sizes as the
    // 0xFFFFFFFF "unknown length" sentinel. hound then rejects the file
    // ("data chunk length is not a multiple of sample size"). Repair the length
    // fields to the real byte counts before parsing.
    let repaired = repair_streaming_wav(wav_bytes);
    let bytes: &[u8] = repaired.as_deref().unwrap_or(wav_bytes);

    let reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| WhisrsError::Audio(format!("failed to read WAV header: {e}")))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| WhisrsError::Audio(format!("failed to decode float WAV samples: {e}")))?,
        hound::SampleFormat::Int => {
            // Normalize integer samples by the full-scale value for the bit depth.
            let max_amp = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_amp))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    WhisrsError::Audio(format!("failed to decode integer WAV samples: {e}"))
                })?
        }
    };

    if spec.channels == 0 {
        return Err(WhisrsError::Audio("WAV reports zero channels".to_string()));
    }

    Ok(DecodedWav {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

/// Repair WAV files written to a non-seekable stream, where the `RIFF` and
/// `data` chunk sizes are left as the 0xFFFFFFFF "unknown length" sentinel (or
/// otherwise overrun the buffer). Rewrites both to the real byte counts.
///
/// Returns `Some(fixed_bytes)` when a repair was applied, or `None` when the
/// input is already well-formed or is not a recognizable RIFF/WAVE stream (in
/// which case the caller passes the original bytes through to the parser).
fn repair_streaming_wav(bytes: &[u8]) -> Option<Vec<u8>> {
    const SENTINEL: u32 = 0xFFFF_FFFF;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let read_u32 =
        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let mut block_align: usize = 0;
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let len = read_u32(off + 4);
        let payload = off + 8;

        if id == b"fmt " && payload + 16 <= bytes.len() {
            // blockAlign sits at offset 12..14 within the fmt payload.
            block_align = u16::from_le_bytes([bytes[payload + 12], bytes[payload + 13]]) as usize;
        }

        if id == b"data" {
            let avail = bytes.len() - payload;
            // Only repair a clearly-bogus length; leave well-formed files alone.
            if len != SENTINEL && (len as usize) <= avail {
                return None;
            }
            let ba = block_align.max(1);
            let fixed = avail - (avail % ba);
            let mut out = bytes.to_vec();
            out[payload - 4..payload].copy_from_slice(&(fixed as u32).to_le_bytes());
            // RIFF size = everything after the leading 8 bytes, through the data payload.
            let riff = (payload + fixed - 8) as u32;
            out[4..8].copy_from_slice(&riff.to_le_bytes());
            return Some(out);
        }

        // Can't safely walk past a sentinel-length chunk that precedes `data`.
        if len == SENTINEL {
            return None;
        }
        off = off.checked_add(8 + len as usize + (len as usize & 1))?;
    }
    None
}

/// Play WAV-encoded audio on the default output device, blocking until the clip
/// finishes or `stop` is set to `true`.
///
/// Builds a cpal output stream matching the WAV's sample rate and channel count.
/// The stream callback advances through the decoded samples; when they are
/// exhausted (or `stop` is set) it emits silence and signals completion. This
/// function is intended to be run on a blocking task (`spawn_blocking`).
///
/// When `level_tx` is `Some`, a normalized amplitude (0..=1) computed from each
/// emitted buffer is published so a speaking overlay can react to the audio.
/// The watch channel coalesces, so no throttling is needed.
pub fn play_wav(
    wav_bytes: &[u8],
    stop: Arc<AtomicBool>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
) -> Result<(), WhisrsError> {
    let decoded = decode_wav(wav_bytes)?;
    play_decoded(decoded, stop, level_tx)
}

type StreamOutput = (
    cpal::Stream,
    SyncSender<Vec<f32>>,
    Arc<AtomicBool>,
    u32,
    u16,
);

/// Incremental playback of PCM16 or IEEE float32 RIFF/WAVE bytes.
///
/// `first_chunk` is consumed before receiving subsequent chunks (including when
/// it is empty). Run on a blocking task. Unsupported WAV formats are buffered and passed to
/// `play_wav` at EOF; once playback starts, an error never replays the clip.
/// The producer must drop its sender to signal EOF. `blocking_recv` cannot
/// observe `stop` while idle: on cancellation, the producer must drop its
/// sender (for example, by selecting on its cancellation signal). `stop` is
/// checked between received chunks and during playback. Do not call from an
/// async runtime worker; use `spawn_blocking`.
/// The output callback never waits for the producer: an empty queue emits silence.
pub fn play_wav_stream(
    first_chunk: Vec<u8>,
    mut receiver: tokio::sync::mpsc::Receiver<Vec<u8>>,
    stop: Arc<AtomicBool>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
) -> Result<(), WhisrsError> {
    let mut parser = StreamWavParser::default();
    let mut buffered = Some(Vec::new());
    let mut output: Option<StreamOutput> = None;
    let mut played_frames = 0usize;
    let mut first_chunk = Some(first_chunk);

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let bytes = match first_chunk.take() {
            Some(bytes) => bytes,
            None => match receiver.blocking_recv() {
                Some(bytes) => bytes,
                None => break,
            },
        };
        if let Some(raw) = &mut buffered {
            raw.extend_from_slice(&bytes);
        }
        if parser.fallback {
            continue;
        }
        let samples = parser.feed(&bytes);
        if parser.fallback {
            if output.is_some() {
                return Err(WhisrsError::Audio(
                    "invalid WAV after playback started".into(),
                ));
            }
            continue;
        }
        if samples.is_empty() {
            continue;
        }
        if output.is_none() {
            let (stream, tx, done, rate, channels) =
                open_stream_output(Arc::clone(&stop), level_tx.clone())?;
            output = Some((stream, tx, done, rate, channels));
            buffered = None;
        }
        let (_, tx, _, rate, channels) = output.as_ref().expect("output opened");
        let fmt = parser.format.expect("samples require fmt");
        let samples = resample_remap(&samples, fmt.channels, fmt.rate, *channels, *rate);
        played_frames += samples.len() / *channels as usize;
        let mut pending = samples;
        loop {
            if stop.load(Ordering::Acquire) {
                break;
            }
            match tx.try_send(pending) {
                Ok(()) => break,
                Err(TrySendError::Full(samples)) => {
                    pending = samples;
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(WhisrsError::Audio("playback stream closed".into()));
                }
            }
        }
    }

    if !stop.load(Ordering::Acquire) {
        if let Some((stream, tx, done, rate, _)) = output {
            if !parser.complete() {
                return Err(WhisrsError::Audio("truncated WAV sample data".into()));
            }
            drop(tx); // the callback drains the queue before marking completion
            let deadline = Instant::now()
                + Duration::from_secs_f64(played_frames as f64 / rate.max(1) as f64 + 2.0);
            while !done.load(Ordering::Acquire) && !stop.load(Ordering::Acquire) {
                if Instant::now() >= deadline {
                    debug!("streaming TTS playback timed out");
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            drop(stream);
        } else if let Some(raw) = buffered {
            if !raw.is_empty() {
                return play_wav(&raw, stop, level_tx);
            }
        }
    }
    if let Some(tx) = &level_tx {
        let _ = tx.send(0.0);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct StreamFormat {
    channels: u16,
    rate: u32,
    bytes_per_sample: usize,
    float: bool,
}

#[derive(Default)]
struct StreamWavParser {
    pending: Vec<u8>,
    format: Option<StreamFormat>,
    header_parsed: bool,
    data_remaining: Option<usize>, // None for sentinel-sized data
    in_data: bool,
    fallback: bool,
}

impl StreamWavParser {
    fn feed(&mut self, bytes: &[u8]) -> Vec<f32> {
        if self.fallback {
            return Vec::new();
        }
        self.pending.extend_from_slice(bytes);
        if !self.in_data {
            if !self.header_parsed {
                if self.pending.len() < 12 {
                    return Vec::new();
                }
                if &self.pending[..4] != b"RIFF" || &self.pending[8..12] != b"WAVE" {
                    self.fallback = true;
                    return Vec::new();
                }
                self.pending.drain(..12);
                self.header_parsed = true;
            }
            let mut offset = 0;
            loop {
                if self.pending.len() - offset < 8 {
                    break;
                }
                let id = &self.pending[offset..offset + 4];
                let len =
                    u32::from_le_bytes(self.pending[offset + 4..offset + 8].try_into().unwrap());
                if id == b"data" {
                    if self.format.is_none() {
                        self.fallback = true;
                        return Vec::new();
                    }
                    let fmt = self.format.expect("format checked above");
                    if len != u32::MAX
                        && !(len as usize)
                            .is_multiple_of(fmt.channels as usize * fmt.bytes_per_sample)
                    {
                        self.fallback = true;
                        return Vec::new();
                    }
                    self.data_remaining = (len != u32::MAX).then_some(len as usize);
                    self.in_data = true;
                    self.pending.drain(..offset + 8);
                    break;
                }
                if len == u32::MAX {
                    self.fallback = true;
                    return Vec::new();
                }
                let end = offset + 8 + len as usize + (len as usize & 1);
                if end > self.pending.len() {
                    break;
                }
                if id == b"fmt " {
                    let fmt = &self.pending[offset + 8..offset + 8 + len as usize];
                    if fmt.len() < 16 {
                        self.fallback = true;
                        return Vec::new();
                    }
                    let encoding = u16::from_le_bytes([fmt[0], fmt[1]]);
                    let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                    let rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                    let align = u16::from_le_bytes([fmt[12], fmt[13]]) as usize;
                    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
                    let bytes_per_sample = (bits / 8) as usize;
                    if channels == 0
                        || rate == 0
                        || align != channels as usize * bytes_per_sample
                        || !matches!((encoding, bits), (1, 16) | (3, 32))
                    {
                        self.fallback = true;
                        return Vec::new();
                    }
                    self.format = Some(StreamFormat {
                        channels,
                        rate,
                        bytes_per_sample,
                        float: encoding == 3,
                    });
                }
                offset = end;
            }
            if !self.in_data {
                self.pending.drain(..offset);
                return Vec::new();
            }
        }
        if self.data_remaining == Some(0) {
            self.pending.clear(); // padding and subsequent RIFF chunks are not audio
            return Vec::new();
        }
        let fmt = self.format.expect("data requires fmt");
        let limit = self
            .data_remaining
            .unwrap_or(self.pending.len())
            .min(self.pending.len());
        let frame_bytes = fmt.bytes_per_sample * fmt.channels as usize;
        let count = limit / frame_bytes * frame_bytes;
        let mut samples = Vec::with_capacity(count / fmt.bytes_per_sample);
        for sample in self.pending[..count].chunks_exact(fmt.bytes_per_sample) {
            samples.push(if fmt.float {
                f32::from_le_bytes(sample.try_into().unwrap())
            } else {
                i16::from_le_bytes(sample.try_into().unwrap()) as f32 / 32768.0
            });
        }
        self.pending.drain(..count);
        if let Some(remaining) = &mut self.data_remaining {
            *remaining -= count;
            if *remaining == 0 {
                self.pending.clear(); // optional data pad and trailing chunks
            }
        }
        samples
    }

    fn complete(&self) -> bool {
        self.in_data && self.data_remaining.is_none_or(|n| n == 0) && self.pending.is_empty()
    }
}

fn open_stream_output(
    stop: Arc<AtomicBool>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
) -> Result<StreamOutput, WhisrsError> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or_else(|| WhisrsError::Audio("no default audio output device".into()))?;
    let cfg = device
        .default_output_config()
        .map_err(|e| WhisrsError::Audio(format!("failed to get default output config: {e}")))?;
    let rate = cfg.sample_rate().0;
    let channels = cfg.channels().max(1);
    let config = StreamConfig {
        channels,
        sample_rate: SampleRate(rate),
        buffer_size: cpal::BufferSize::Default,
    };
    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(8);
    let done = Arc::new(AtomicBool::new(false));
    let done_cb = Arc::clone(&done);
    let mut current = Vec::new();
    let mut index = 0;
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                for sample in data.iter_mut() {
                    if stop.load(Ordering::Acquire) {
                        *sample = 0.0;
                        continue;
                    }
                    if index == current.len() {
                        match rx.try_recv() {
                            Ok(next) => {
                                current = next;
                                index = 0;
                            }
                            Err(TryRecvError::Disconnected) => {
                                done_cb.store(true, Ordering::Release);
                            }
                            Err(TryRecvError::Empty) => {}
                        }
                    }
                    *sample = if index < current.len() {
                        let value = current[index];
                        index += 1;
                        value
                    } else {
                        0.0
                    };
                }
                if let Some(tx) = &level_tx {
                    let _ = tx.send(playback_level(data));
                }
            },
            |err| warn!("TTS playback stream error: {err}"),
            None,
        )
        .map_err(|e| WhisrsError::Audio(format!("failed to build output stream: {e}")))?;
    stream
        .play()
        .map_err(|e| WhisrsError::Audio(format!("failed to start playback: {e}")))?;
    Ok((stream, tx, done, rate, channels))
}

/// Normalized playback amplitude from a buffer of interleaved f32 samples.
///
/// Mirrors [`crate::audio::capture`]'s level mapping: RMS through a soft
/// compressor `1 - exp(-rms * 18)` so typical speech reaches the upper part
/// of the visualizer's dynamic range.
fn playback_level(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = data.iter().map(|s| s * s).sum();
    let rms = (sum_squares / data.len() as f32).sqrt();
    (1.0 - (-rms * 18.0).exp()).clamp(0.0, 1.0)
}

/// Resample interleaved f32 audio from `(src_rate, src_channels)` to
/// `(dst_rate, dst_channels)`.
///
/// The source is downmixed to mono and then fanned out across the destination
/// channels, with linear interpolation for the rate conversion. This is aimed
/// at speech (Groq's TTS is mono); a stereo music source would lose its imaging,
/// which is acceptable for this playback path.
fn resample_remap(
    src: &[f32],
    src_channels: u16,
    src_rate: u32,
    dst_channels: u16,
    dst_rate: u32,
) -> Vec<f32> {
    let src_ch = src_channels.max(1) as usize;
    let dst_ch = dst_channels.max(1) as usize;
    let src_frames = src.len() / src_ch;
    if src_frames == 0 || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }

    // Downmix to a mono signal.
    let mono: Vec<f32> = (0..src_frames)
        .map(|i| {
            let acc: f32 = (0..src_ch).map(|c| src[i * src_ch + c]).sum();
            acc / src_ch as f32
        })
        .collect();

    // Fan a mono frame value out across all destination channels.
    let fan = |out: &mut Vec<f32>, v: f32| {
        for _ in 0..dst_ch {
            out.push(v);
        }
    };

    if src_rate == dst_rate {
        let mut out = Vec::with_capacity(mono.len() * dst_ch);
        for v in mono {
            fan(&mut out, v);
        }
        return out;
    }

    let dst_frames = ((src_frames as u64 * dst_rate as u64) / src_rate as u64).max(1) as usize;
    let ratio = src_rate as f64 / dst_rate as f64;
    let mut out = Vec::with_capacity(dst_frames * dst_ch);
    for j in 0..dst_frames {
        let pos = j as f64 * ratio;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = mono[i0.min(src_frames - 1)];
        let b = mono[(i0 + 1).min(src_frames - 1)];
        fan(&mut out, a + (b - a) * frac);
    }
    out
}

/// Play already-decoded PCM on the default output device. See [`play_wav`].
fn play_decoded(
    decoded: DecodedWav,
    stop: Arc<AtomicBool>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
) -> Result<(), WhisrsError> {
    if decoded.samples.is_empty() {
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| WhisrsError::Audio("no default audio output device".to_string()))?;

    // Play at the device's preferred rate/channels and resample/remap the clip
    // to match. Forcing the clip's native format (e.g. Groq's 24 kHz mono) can
    // fail to build a stream on devices that only advertise their default rate.
    let (target_rate, target_channels) = match device.default_output_config() {
        Ok(cfg) => (cfg.sample_rate().0, cfg.channels().max(1)),
        Err(e) => {
            debug!("no default output config ({e}); using clip's native format");
            (decoded.sample_rate, decoded.channels.max(1))
        }
    };

    let config = StreamConfig {
        channels: target_channels,
        sample_rate: SampleRate(target_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let samples = resample_remap(
        &decoded.samples,
        decoded.channels,
        decoded.sample_rate,
        target_channels,
        target_rate,
    );
    if samples.is_empty() {
        return Ok(());
    }
    let samples_len = samples.len();
    let sample_idx = Arc::new(AtomicUsize::new(0));
    let sample_idx_cb = Arc::clone(&sample_idx);
    let done = Arc::new(AtomicBool::new(false));
    let done_cb = Arc::clone(&done);
    let stop_cb = Arc::clone(&stop);

    let level_tx_cb = level_tx.clone();
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                if stop_cb.load(Ordering::Acquire) {
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    done_cb.store(true, Ordering::Release);
                    if let Some(tx) = &level_tx_cb {
                        let _ = tx.send(0.0);
                    }
                    return;
                }
                for sample in data.iter_mut() {
                    let idx = sample_idx_cb.fetch_add(1, Ordering::Relaxed);
                    if idx < samples_len {
                        *sample = samples[idx];
                    } else {
                        *sample = 0.0;
                        done_cb.store(true, Ordering::Release);
                    }
                }
                // Publish the amplitude of the buffer we just emitted so a
                // speaking overlay can react. Best-effort; watch coalesces.
                if let Some(tx) = &level_tx_cb {
                    let _ = tx.send(playback_level(data));
                }
            },
            |err| {
                warn!("TTS playback stream error: {err}");
            },
            None,
        )
        .map_err(|e| WhisrsError::Audio(format!("failed to build output stream: {e}")))?;

    stream
        .play()
        .map_err(|e| WhisrsError::Audio(format!("failed to start playback: {e}")))?;

    // Compute a generous upper bound on playback duration so a stuck stream
    // can't block forever.
    let frames = (samples_len / target_channels.max(1) as usize) as f64;
    let clip_secs = frames / target_rate.max(1) as f64;
    let timeout = Duration::from_secs_f64(clip_secs + 2.0);
    let start = Instant::now();

    while !done.load(Ordering::Acquire) {
        if stop.load(Ordering::Acquire) {
            debug!("TTS playback interrupted");
            break;
        }
        if start.elapsed() > timeout {
            debug!("TTS playback timed out after {:.1}s", timeout.as_secs_f64());
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    // Let the final buffer drain.
    std::thread::sleep(Duration::from_millis(50));
    drop(stream);
    // Reset the visualizer to silence once playback ends.
    if let Some(tx) = &level_tx {
        let _ = tx.send(0.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a short stereo i16 tone WAV in memory for decode round-trip tests.
    fn make_i16_wav(sample_rate: u32, channels: u16, frames: usize) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            for i in 0..frames {
                let v = ((i as f32 * 0.01).sin() * i16::MAX as f32) as i16;
                for _ in 0..channels {
                    writer.write_sample(v).unwrap();
                }
            }
            writer.finalize().unwrap();
        }
        buf.into_inner()
    }

    fn make_f32_wav(sample_rate: u32, channels: u16, frames: usize) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            for i in 0..frames {
                let v = (i as f32 * 0.01).sin() * 0.5;
                for _ in 0..channels {
                    writer.write_sample(v).unwrap();
                }
            }
            writer.finalize().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn decode_i16_stereo_wav() {
        let wav = make_i16_wav(24_000, 2, 100);
        let decoded = decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples.len(), 200);
        assert_eq!(decoded.frames(), 100);
        // Normalized into [-1.0, 1.0].
        assert!(decoded.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
    }

    #[test]
    fn decode_i16_mono_wav() {
        let wav = make_i16_wav(16_000, 1, 50);
        let decoded = decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), 50);
        assert_eq!(decoded.frames(), 50);
    }

    #[test]
    fn decode_f32_wav() {
        let wav = make_f32_wav(44_100, 1, 64);
        let decoded = decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), 64);
    }

    #[test]
    fn decode_rejects_garbage() {
        let err = decode_wav(b"not a wav file at all").unwrap_err();
        assert!(err.to_string().contains("WAV header"));
    }

    /// A WAV streamed by ffmpeg/Lavf (as Groq returns) leaves the RIFF and data
    /// chunk lengths as the 0xFFFFFFFF sentinel; hound rejects it raw, but
    /// `decode_wav` repairs the lengths first.
    #[test]
    fn decode_repairs_streaming_sentinel_lengths() {
        let mut wav = make_i16_wav(24_000, 1, 100);
        let dpos = wav
            .windows(4)
            .position(|w| w == b"data")
            .expect("data chunk header");
        wav[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // RIFF size sentinel
        wav[dpos + 4..dpos + 8].copy_from_slice(&u32::MAX.to_le_bytes()); // data size sentinel

        // Raw hound rejects the sentinel data length...
        assert!(hound::WavReader::new(Cursor::new(&wav)).is_err());
        // ...but decode_wav repairs and reads it.
        let decoded = decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.frames(), 100);
    }

    #[test]
    fn stream_receiver_closed_or_cancelled_before_audio() {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        drop(tx);
        play_wav_stream(Vec::new(), rx, Arc::new(AtomicBool::new(false)), None).unwrap();

        let (_tx, rx) = tokio::sync::mpsc::channel(2);
        play_wav_stream(vec![1, 2, 3], rx, Arc::new(AtomicBool::new(true)), None).unwrap();

        // Dropping the sender wakes a playback thread blocked in blocking_recv.
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let playback = std::thread::spawn(move || {
            play_wav_stream(Vec::new(), rx, Arc::new(AtomicBool::new(false)), None)
        });
        drop(tx);
        playback.join().unwrap().unwrap();
    }

    #[test]
    fn stream_first_chunk_precedes_receiver_even_if_empty() {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        drop(tx);
        assert!(play_wav_stream(
            b"invalid WAV".to_vec(),
            rx,
            Arc::new(AtomicBool::new(false)),
            None
        )
        .is_err());

        let (tx, rx) = tokio::sync::mpsc::channel(2);
        tx.blocking_send(b"invalid WAV".to_vec()).unwrap();
        drop(tx);
        assert!(play_wav_stream(Vec::new(), rx, Arc::new(AtomicBool::new(false)), None).is_err());
    }

    #[test]
    fn stream_parser_split_headers_samples_and_padded_chunks() {
        let wav = make_i16_wav(24_000, 1, 3);
        let mut with_junk = wav[..12].to_vec();
        with_junk.extend_from_slice(b"JUNK");
        with_junk.extend_from_slice(&3u32.to_le_bytes());
        with_junk.extend_from_slice(&[1, 2, 3, 0]);
        with_junk.extend_from_slice(&wav[12..]);
        let mut parser = StreamWavParser::default();
        let mut samples = Vec::new();
        for byte in with_junk.chunks(1) {
            samples.extend(parser.feed(byte));
        }
        assert!(!parser.fallback);
        assert!(parser.complete());
        assert_eq!(samples, decode_wav(&wav).unwrap().samples);
    }

    #[test]
    fn stream_parser_sentinel_and_trailing_pad() {
        let mut wav = make_i16_wav(24_000, 1, 3);
        let data = wav.windows(4).position(|w| w == b"data").unwrap();
        wav[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        wav[data + 4..data + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut parser = StreamWavParser::default();
        let mut samples = Vec::new();
        for piece in wav.chunks(5) {
            samples.extend(parser.feed(piece));
        }
        assert_eq!(samples, decode_wav(&wav).unwrap().samples);
        assert!(parser.complete());

        let mut finite = make_i16_wav(24_000, 1, 3);
        finite.extend_from_slice(b"JUNK\0\0\0\0");
        let mut parser = StreamWavParser::default();
        assert_eq!(parser.feed(&finite).len(), 3);
        assert!(parser.complete());
        assert_eq!(parser.feed(b"trailing bytes"), Vec::<f32>::new());
    }

    #[test]
    fn stream_parser_unsupported_format_falls_back_before_audio() {
        let mut wav = make_i16_wav(24_000, 1, 3);
        let fmt = wav.windows(4).position(|w| w == b"fmt ").unwrap();
        wav[fmt + 8..fmt + 10].copy_from_slice(&6u16.to_le_bytes());
        let mut parser = StreamWavParser::default();
        for piece in wav.chunks(7) {
            assert!(parser.feed(piece).is_empty());
        }
        assert!(parser.fallback);

        // A valid format that hound supports still takes the buffered path.
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 8,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            writer.write_sample(42i8).unwrap();
            writer.finalize().unwrap();
        }
        let wav8 = cursor.into_inner();
        let mut parser = StreamWavParser::default();
        assert!(parser.feed(&wav8).is_empty());
        assert!(parser.fallback);
        assert_eq!(decode_wav(&wav8).unwrap().frames(), 1);

        let mut truncated = StreamWavParser::default();
        let wav = make_i16_wav(24_000, 1, 3);
        truncated.feed(&wav[..wav.len() - 1]);
        assert!(!truncated.complete());
    }

    #[test]
    fn resample_noop_when_format_matches() {
        let mono = vec![0.1f32, 0.2, 0.3, 0.4];
        assert_eq!(resample_remap(&mono, 1, 16_000, 1, 16_000), mono);
    }

    #[test]
    fn resample_changes_rate_and_upmixes_channels() {
        // 24 kHz mono -> 48 kHz stereo: frame count doubles, channels double.
        let mono: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = resample_remap(&mono, 1, 24_000, 2, 48_000);
        assert_eq!(out.len(), 200 * 2);
        // Left/right are identical (mono fanned out).
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn playback_level_silence_is_zero() {
        assert_eq!(playback_level(&[]), 0.0);
        assert_eq!(playback_level(&[0.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn playback_level_loud_is_higher_than_quiet() {
        let quiet = playback_level(&[0.01, -0.01, 0.01, -0.01]);
        let loud = playback_level(&[0.8, -0.8, 0.8, -0.8]);
        assert!(loud > quiet);
        assert!((0.0..=1.0).contains(&loud));
        assert!((0.0..=1.0).contains(&quiet));
    }

    #[test]
    fn resample_downmix_stereo_to_mono() {
        // Interleaved stereo [1.0, -1.0, ...] downmixes to ~0.0 mono, same rate.
        let stereo = vec![1.0f32, -1.0, 1.0, -1.0];
        let out = resample_remap(&stereo, 2, 16_000, 1, 16_000);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|s| s.abs() < 1e-6));
    }
}
