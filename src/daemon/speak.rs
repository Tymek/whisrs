use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use whisrs::state::Action;
use whisrs::{Response, State};

use crate::context::{DaemonContext, DaemonState};
use crate::factory::resolve_tts_api_key;
use crate::notify::send_notification;
use crate::selection::capture_selection;

/// Bounded wait for the first TTS audio chunk before failing the read-aloud.
///
/// Streaming time-to-first-audio is on the order of ~100 ms, so this is
/// generous; it only fires when the TTS server is stalled and would otherwise
/// leave the daemon stuck in `Synthesizing` forever waiting for a chunk that
/// never arrives (e.g. a server deadlocked on its own generation lock).
const FIRST_CHUNK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long to wait for the producer task to unwind after signalling `stop` on
/// a first-chunk timeout before giving up on awaiting it.
const PRODUCER_SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// Read the selected text aloud via TTS.
///
/// FSM-authoritative: read-aloud has its own `Synthesizing`/`Speaking` states,
/// so "is read-aloud active" is determined by [`State`], not the `tts_stop`
/// flag (which now exists purely as the low-level playback interrupt). This
/// removes the race where a `Speak` press landing in the gap between playback
/// finishing and the cleanup task clearing `tts_stop` was swallowed as a stop.
///
/// The daemon mutex is never held across the synth or playback `.await`.
pub(crate) async fn handle_speak(
    daemon_state: Arc<Mutex<DaemonState>>,
    context: Arc<DaemonContext>,
) -> Response {
    use std::sync::atomic::{AtomicBool, Ordering};

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
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut ds = daemon_state.lock().await;
        if ds.state_machine.state() != State::Idle {
            return Response::Ok {
                state: ds.state_machine.state(),
            };
        }
        ds.tts_stop = Some(Arc::clone(&stop));
        let _ = ds.state_machine.transition(Action::SpeakStart);
        let _ = context.state_tx.send(ds.state_machine.state());
    }

    info!("speak: synthesizing {} chars", selected_text.len());
    if context.notify_state() {
        send_notification("whisrs", "Reading selection aloud...");
    }

    // (e) Start a bounded producer and wait only for the first body chunk.
    // Waiting here keeps HTTP errors visible to the caller and permits a second
    // Speak to cancel synthesis before playback begins.
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    let producer_stop = Arc::clone(&stop);
    let producer = tokio::spawn(async move {
        tokio::select! {
            result = backend.synthesize_stream(&selected_text, sender, Arc::clone(&producer_stop)) => result,
            _ = async {
                while !producer_stop.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            } => Ok(()),
        }
    });
    // (e2) Wait a bounded time for the first chunk. A stalled TTS server that
    // never opens its response would otherwise hang the daemon in `Synthesizing`
    // indefinitely; timeout and surface an error instead.
    let first_chunk = tokio::time::timeout(FIRST_CHUNK_TIMEOUT, receiver.recv()).await;

    // (f) Re-lock to decide whether to play.
    let final_state = {
        let mut ds = daemon_state.lock().await;

        // A second press during synthesis cancelled us — state is no longer
        // Synthesizing. Don't play; report the current state.
        if ds.state_machine.state() != State::Synthesizing {
            info!("speak: synthesis superseded by a concurrent command");
            return Response::Ok {
                state: ds.state_machine.state(),
            };
        }

        let first_chunk = match first_chunk {
            // A chunk arrived in time.
            Ok(Some(bytes)) => bytes,
            // Producer closed without a first chunk (HTTP/body error) or the
            // server stalled past the first-chunk timeout. Either way: cancel
            // the producer, then finalize the FSM back to Idle with an error.
            outcome => {
                drop(ds);
                let message = match outcome {
                    Ok(None) => match producer.await {
                        Ok(Err(e)) => e.to_string(),
                        Ok(Ok(())) => "TTS returned an empty response".to_string(),
                        Err(e) => format!("TTS task failed: {e}"),
                    },
                    Err(_) => {
                        stop.store(true, Ordering::Release);
                        let _ = tokio::time::timeout(PRODUCER_SHUTDOWN_GRACE, producer).await;
                        "TTS timed out before producing audio".to_string()
                    }
                    Ok(Some(_)) => unreachable!(),
                };
                let mut ds = daemon_state.lock().await;
                if ds.state_machine.state() != State::Synthesizing
                    || !ds.tts_stop.as_ref().is_some_and(|s| Arc::ptr_eq(s, &stop))
                {
                    return Response::Ok {
                        state: ds.state_machine.state(),
                    };
                }
                error!("speak: synthesis failed: {message}");
                ds.tts_stop = None;
                let _ = ds.state_machine.transition(Action::SpeakDone);
                let _ = context.state_tx.send(ds.state_machine.state());
                if let Some(level_tx) = &context.overlay_level_tx {
                    let _ = level_tx.send(0.0);
                }
                drop(ds);
                if context.notify_error() {
                    send_notification("whisrs", &format!("Read-aloud failed: {message}"));
                }
                return Response::Error {
                    message: format!("TTS synthesis failed: {message}"),
                };
            }
        };
        let _ = ds.state_machine.transition(Action::SpeakPlaying);
        let new_state = ds.state_machine.state();
        let _ = context.state_tx.send(new_state);

        // (g) Spawn interruptible playback feeding the speaking overlay.
        let ds_ref = Arc::clone(&daemon_state);
        let context_for_cleanup = Arc::clone(&context);
        let playback_stop = Arc::clone(&stop);
        let level_tx = context.overlay_level_tx.clone();
        tokio::spawn(async move {
            let playback = tokio::task::spawn_blocking(move || {
                whisrs::audio::playback::play_wav_stream(first_chunk, receiver, stop, level_tx)
            });
            // Playback drains concurrently with synthesis, not after it.
            let result = playback.await;
            match producer.await {
                Ok(Err(e)) => warn!("speak: TTS stream failed: {e}"),
                Err(e) => warn!("speak: TTS task panicked: {e}"),
                _ => {}
            }
            match result {
                Ok(Ok(())) => debug!("speak: playback finished"),
                Ok(Err(e)) => warn!("speak: playback failed: {e}"),
                Err(e) => warn!("speak: playback task panicked: {e}"),
            }

            // On finish/interrupt: only finalize if we still own playback and
            // the FSM is still Speaking. A second Speak / Cancel already
            // transitioned to Idle and replaced/cleared tts_stop.
            let mut ds = ds_ref.lock().await;
            let still_ours = ds
                .tts_stop
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &playback_stop));
            if ds.state_machine.state() == State::Speaking && still_ours {
                let _ = ds.state_machine.transition(Action::SpeakDone);
                ds.tts_stop = None;
                let _ = context_for_cleanup.state_tx.send(ds.state_machine.state());
                if let Some(level_tx) = &context_for_cleanup.overlay_level_tx {
                    let _ = level_tx.send(0.0);
                }
            }
        });

        new_state
    };

    Response::Ok { state: final_state }
}
