use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use whisrs::audio::playback::{play_wav_stream, StreamChunk};
use whisrs::state::Action;
use whisrs::tts::TtsAudioStream;
use whisrs::{Response, State};

use crate::context::{DaemonContext, DaemonState};
use crate::factory::resolve_tts_api_key;
use crate::notify::send_notification;
use crate::selection::capture_selection;

/// Read the selected text aloud via TTS.
///
/// FSM-authoritative: read-aloud has its own `Synthesizing`/`Speaking` states,
/// so "is read-aloud active" is determined by [`State`], not the `tts_stop`
/// flag (which now exists purely as the low-level playback interrupt). This
/// removes the race where a `Speak` press landing in the gap between playback
/// finishing and the cleanup task clearing `tts_stop` was swallowed as a stop.
///
/// Playback is incremental: the state stays `Synthesizing` until the first
/// samples of the streamed body are queued, then moves to `Speaking`, and the
/// command replies at that point rather than after the whole clip downloads.
///
/// The daemon mutex is never held across the synth, chunk-read or playback
/// `.await`.
pub(crate) async fn handle_speak(
    daemon_state: Arc<Mutex<DaemonState>>,
    context: Arc<DaemonContext>,
) -> Response {
    // (a) Inspect the FSM. A second press while read-aloud is active is a
    // deterministic stop, regardless of playback timing.
    {
        let mut ds = daemon_state.lock().await;
        match ds.state_machine.state() {
            State::Speaking | State::Synthesizing => {
                if let Some(stop) = ds.tts_stop.take() {
                    stop.store(true, Ordering::Release);
                }
                let _ = ds.state_machine.transition(Action::Cancel);
                let new_state = ds.state_machine.state();
                let _ = context.state_tx.send(new_state);
                if let Some(level_tx) = &context.overlay_level_tx {
                    let _ = level_tx.send(0.0);
                }
                info!("speak: stopped in-progress read-aloud");
                return Response::Ok { state: new_state };
            }
            State::Recording | State::Transcribing => {
                return Response::Error {
                    message: "busy — finish or cancel recording first".to_string(),
                };
            }
            State::Idle => {
                // Fall through; do NOT transition yet.
            }
        }
    }

    // (b) Verify TTS is configured/enabled and build the backend. State is
    // still Idle on any error here — nothing to unwind.
    let tts_config = match &context.config.tts {
        Some(t) if t.enabled => t.clone(),
        Some(_) => {
            return Response::Error {
                message: "TTS is disabled — set [tts] enabled = true in config.toml".to_string(),
            };
        }
        None => {
            return Response::Error {
                message: "TTS is not configured — add a [tts] section to config.toml".to_string(),
            };
        }
    };

    let api_key = resolve_tts_api_key(&context.config);
    let backend = match whisrs::tts::create_backend(&tts_config, api_key) {
        Ok(b) => b,
        Err(e) => {
            return Response::Error {
                message: e.to_string(),
            };
        }
    };

    // (c) Capture the selection. On failure, surface an error and — since this
    // is hotkey-triggered — notify so the user sees why.
    //
    // Every `CaptureError` variant is fatal here, deliberately: read-aloud has
    // nothing to say without text. That includes both "nothing was selected"
    // variants — `NothingSelected` (the clipboard came back empty) and
    // `ClipboardUnchanged` (the capture could not tell). Only a caller that
    // would proceed on an empty selection needs to tell them apart, and
    // read-aloud never proceeds, so it just renders the variant's `Display`,
    // whose wording is byte-identical across all of them and unchanged from
    // before the error was typed.
    info!("speak: getting selected text");
    let selected_text = match capture_selection(&context).await {
        Ok(text) => text,
        Err(error) => {
            let message = error.to_string();
            if context.notify_error() {
                send_notification("whisrs", &format!("Read-aloud: {message}"));
            }
            return Response::Error { message };
        }
    };

    // (d) Re-lock; only begin synthesizing if still Idle (a concurrent command
    // may have intervened). Do NOT proceed otherwise.
    //
    // `tts_stop` is installed here, not when audio starts, so a second press or
    // `whisrs cancel` during synthesis also interrupts the stream reader and
    // playback. Its identity (`Arc::ptr_eq`) marks this session as the owner.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut ds = daemon_state.lock().await;
        if ds.state_machine.state() != State::Idle {
            return Response::Ok {
                state: ds.state_machine.state(),
            };
        }
        let _ = ds.state_machine.transition(Action::SpeakStart);
        ds.tts_stop = Some(Arc::clone(&stop));
        let _ = context.state_tx.send(ds.state_machine.state());
    }

    info!("speak: synthesizing {} chars", selected_text.len());
    if context.notify_state() {
        send_notification("whisrs", "Reading selection aloud...");
    }
    let started_at = Instant::now();

    // (e) Start synthesis (no lock held). This returns once the response
    // headers arrive; the audio body is streamed afterwards. It is raced
    // against `stop`, so a second press / cancel while a slow server holds
    // the headers drops the request instead of leaving it running.
    let synth_result = until_stopped(backend.synthesize(&selected_text), &stop).await;

    // (f) Re-lock to decide whether to play.
    let audio = {
        let mut ds = daemon_state.lock().await;

        let Some(synth_result) = synth_result else {
            info!("speak: synthesis cancelled before the response arrived");
            return Response::Ok {
                state: ds.state_machine.state(),
            };
        };

        // A second press / cancel during synthesis stopped us — state is no
        // longer ours. Don't play; report the current state. Dropping the
        // stream closes the connection.
        if ds.state_machine.state() != State::Synthesizing || !owns(&ds, &stop) {
            info!("speak: synthesis superseded by a concurrent command");
            return Response::Ok {
                state: ds.state_machine.state(),
            };
        }

        match synth_result {
            Ok(audio) => audio,
            Err(e) => {
                error!("speak: synthesis failed: {e}");
                finish_session(&mut ds, &context);
                drop(ds);
                let (notification, message) = synthesis_failure_messages(e);
                if context.notify_error() {
                    send_notification("whisrs", &notification);
                }
                return Response::Error { message };
            }
        }
    };

    // (g) Stream the body into incremental playback: an async reader forwards
    // chunks over a std channel to the blocking player.
    let (chunk_tx, chunk_rx) = std::sync::mpsc::channel();
    tokio::spawn(forward_chunks(audio, chunk_tx, Arc::clone(&stop)));

    let (first_audio_tx, first_audio_rx) = tokio::sync::oneshot::channel();
    let level_tx = context.overlay_level_tx.clone();
    let playback_stop = Arc::clone(&stop);
    let playback = tokio::task::spawn_blocking(move || {
        play_wav_stream(chunk_rx, playback_stop, level_tx, first_audio_tx)
    });

    // (h) Stay in Synthesizing until the first samples are queued. The player
    // owns the sender, so the receiver also resolves (as Err) when playback
    // ends, errors or panics before producing any audio.
    if first_audio_rx.await.is_err() {
        let result = playback.await;
        // Unblock the reader if it is still waiting on the network.
        stop.store(true, Ordering::Release);
        debug!(
            "speak: playback ended before any audio after {:?}",
            started_at.elapsed()
        );
        let failure = playback_failure(result);

        let mut ds = daemon_state.lock().await;
        if ds.state_machine.state() != State::Synthesizing || !owns(&ds, &stop) {
            info!("speak: read-aloud stopped before audio started");
            return Response::Ok {
                state: ds.state_machine.state(),
            };
        }
        finish_session(&mut ds, &context);
        let state = ds.state_machine.state();
        drop(ds);
        return match failure {
            None => {
                // A well-formed body that held no samples: nothing to say.
                warn!("speak: TTS returned no audio");
                Response::Ok { state }
            }
            Some(e) => {
                error!("speak: playback failed before audio started: {e}");
                if context.notify_error() {
                    send_notification("whisrs", &format!("Read-aloud failed: {e}"));
                }
                Response::Error {
                    message: format!("TTS playback failed: {e}"),
                }
            }
        };
    }

    debug!("speak: first audio after {:?}", started_at.elapsed());

    let final_state = {
        let mut ds = daemon_state.lock().await;
        if ds.state_machine.state() == State::Synthesizing && owns(&ds, &stop) {
            let _ = ds.state_machine.transition(Action::SpeakPlaying);
            let _ = context.state_tx.send(ds.state_machine.state());
        } else {
            // Cancelled in the window between first audio and this lock; the
            // canceller already set `stop`, so playback is winding down.
            info!("speak: read-aloud stopped as audio started");
        }
        ds.state_machine.state()
    };

    // (i) Finalize once playback ends. Only touch the FSM if this session
    // still owns it: a second Speak / Cancel already moved to Idle and
    // cleared (or replaced) `tts_stop`.
    let ds_ref = Arc::clone(&daemon_state);
    tokio::spawn(async move {
        let result = playback.await;
        stop.store(true, Ordering::Release);
        debug!("speak: playback ended after {:?}", started_at.elapsed());
        let failure = playback_failure(result);

        let mut ds = ds_ref.lock().await;
        let active = matches!(
            ds.state_machine.state(),
            State::Speaking | State::Synthesizing
        );
        if !(active && owns(&ds, &stop)) {
            return;
        }
        finish_session(&mut ds, &context);
        drop(ds);
        if let Some(e) = failure {
            // Mid-stream network failure, idle timeout or decode error.
            warn!("speak: playback failed: {e}");
            if context.notify_error() {
                send_notification("whisrs", &format!("Read-aloud failed: {e}"));
            }
        }
    });

    Response::Ok { state: final_state }
}

/// Whether `stop` is still the daemon's current read-aloud interrupt, i.e.
/// this session has not been cancelled or replaced.
fn owns(ds: &DaemonState, stop: &Arc<AtomicBool>) -> bool {
    ds.tts_stop
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, stop))
}

/// End the read-aloud session (from Synthesizing or Speaking) and reset the
/// overlay. The caller has checked that it owns the session.
fn finish_session(ds: &mut DaemonState, context: &DaemonContext) {
    let _ = ds.state_machine.transition(Action::SpeakDone);
    ds.tts_stop = None;
    let _ = context.state_tx.send(ds.state_machine.state());
    if let Some(level_tx) = &context.overlay_level_tx {
        let _ = level_tx.send(0.0);
    }
}

/// Collapse the playback task's outcome into an error message, if any.
///
/// The message is the error's own text without the `WhisrsError` category
/// prefix, since the caller already frames it ("Read-aloud failed: ...",
/// "TTS playback failed: ...").
fn playback_failure(
    result: Result<Result<(), whisrs::WhisrsError>, tokio::task::JoinError>,
) -> Option<String> {
    match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(inner_message(e)),
        Err(e) => Some(format!("playback task panicked: {e}")),
    }
}

/// The message carried by an audio / transcription error, without the
/// "audio error: " / "transcription error: " prefix its `Display` adds.
fn inner_message(e: whisrs::WhisrsError) -> String {
    match e {
        whisrs::WhisrsError::Audio(m) | whisrs::WhisrsError::Transcription(m) => m,
        other => other.to_string(),
    }
}

/// The notification and IPC error text for a failed synthesis request, framed
/// like the playback failures (no nested "transcription error: " prefix).
fn synthesis_failure_messages(e: whisrs::WhisrsError) -> (String, String) {
    let msg = inner_message(e);
    (
        format!("Read-aloud failed: {msg}"),
        format!("TTS synthesis failed: {msg}"),
    )
}

/// How often `stop` is re-checked while waiting on the network.
const STOP_POLL: Duration = Duration::from_millis(50);

/// Resolve once `stop` is set, polling every [`STOP_POLL`].
async fn stopped(stop: &AtomicBool) {
    while !stop.load(Ordering::Acquire) {
        tokio::time::sleep(STOP_POLL).await;
    }
}

/// Run `fut` until it completes or `stop` is set. On stop the future is
/// dropped (closing any connection it holds) and `None` is returned.
async fn until_stopped<F: std::future::Future>(fut: F, stop: &AtomicBool) -> Option<F::Output> {
    tokio::select! {
        out = fut => Some(out),
        _ = stopped(stop) => None,
    }
}

/// Forward the TTS body to the player until it ends, fails, the player hangs
/// up, or `stop` is set. `stop` is also polled while a read is pending, so a
/// stalled connection is dropped promptly on cancel instead of lingering.
async fn forward_chunks(
    mut audio: Box<dyn TtsAudioStream>,
    tx: std::sync::mpsc::Sender<StreamChunk>,
    stop: Arc<AtomicBool>,
) {
    loop {
        // On stop, return without an end marker: the player checks `stop`
        // when the channel disconnects and reports an interruption.
        let Some(next) = until_stopped(audio.next_chunk(), &stop).await else {
            return;
        };
        let (chunk, last) = match next {
            Ok(Some(data)) => (StreamChunk::Data(data), false),
            Ok(None) => (StreamChunk::End, true),
            Err(e) => (StreamChunk::Error(inner_message(e)), true),
        };
        if tx.send(chunk).is_err() || last {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sets its flag when dropped, to observe that a raced future is dropped.
    struct DropFlag(Arc<AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn until_stopped_drops_a_pending_future_on_stop() {
        let stop = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&dropped));
        // Stands in for a synth request whose server never sends headers.
        let pending = async move {
            let _guard = guard;
            std::future::pending::<()>().await
        };
        let setter = {
            let stop = Arc::clone(&stop);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(120)).await;
                stop.store(true, Ordering::Release);
            })
        };
        let started = tokio::time::Instant::now();
        assert!(until_stopped(pending, &stop).await.is_none());
        assert!(dropped.load(Ordering::Acquire), "future must be dropped");
        assert!(started.elapsed() <= Duration::from_millis(120) + STOP_POLL);
        setter.await.unwrap();
    }

    #[tokio::test]
    async fn until_stopped_returns_the_output_when_not_stopped() {
        let stop = AtomicBool::new(false);
        assert_eq!(until_stopped(async { 7 }, &stop).await, Some(7));
    }

    #[test]
    fn failure_messages_drop_the_error_category_prefix() {
        let e = whisrs::WhisrsError::Transcription("TTS read body failed: reset".into());
        assert_eq!(inner_message(e), "TTS read body failed: reset");
        let e = whisrs::WhisrsError::Audio("no default audio output device".into());
        assert_eq!(
            playback_failure(Ok(Err(e))).as_deref(),
            Some("no default audio output device")
        );
        assert_eq!(playback_failure(Ok(Ok(()))), None);
    }

    #[test]
    fn synthesis_failure_messages_drop_the_error_category_prefix() {
        let e = whisrs::WhisrsError::Transcription("TTS error (503): overloaded".into());
        let (notification, message) = synthesis_failure_messages(e);
        assert_eq!(
            notification,
            "Read-aloud failed: TTS error (503): overloaded"
        );
        assert_eq!(message, "TTS synthesis failed: TTS error (503): overloaded");
    }

    #[tokio::test]
    async fn body_error_reaches_the_player_without_nested_prefixes() {
        struct FailingBody;
        #[async_trait::async_trait]
        impl TtsAudioStream for FailingBody {
            async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, whisrs::WhisrsError> {
                Err(whisrs::WhisrsError::Transcription(
                    "TTS read body failed: reset".into(),
                ))
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        forward_chunks(Box::new(FailingBody), tx, Arc::new(AtomicBool::new(false))).await;
        match rx.recv().unwrap() {
            StreamChunk::Error(msg) => assert_eq!(msg, "TTS read body failed: reset"),
            other => panic!("expected an error chunk, got {other:?}"),
        }
    }
}
