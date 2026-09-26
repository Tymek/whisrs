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
    // headers arrive; the audio body is streamed afterwards.
    let synth_result = backend.synthesize(&selected_text).await;

    // (f) Re-lock to decide whether to play.
    let audio = {
        let mut ds = daemon_state.lock().await;

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
                if context.notify_error() {
                    send_notification("whisrs", &format!("Read-aloud failed: {e}"));
                }
                return Response::Error {
                    message: format!("TTS synthesis failed: {e}"),
                };
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
fn playback_failure(
    result: Result<Result<(), whisrs::WhisrsError>, tokio::task::JoinError>,
) -> Option<String> {
    match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e.to_string()),
        Err(e) => Some(format!("playback task panicked: {e}")),
    }
}

/// How often the chunk reader re-checks `stop` while waiting on the network.
const STOP_POLL: Duration = Duration::from_millis(50);

/// Forward the TTS body to the player until it ends, fails, the player hangs
/// up, or `stop` is set. `stop` is also polled while a read is pending, so a
/// stalled connection is dropped promptly on cancel instead of lingering.
async fn forward_chunks(
    mut audio: Box<dyn TtsAudioStream>,
    tx: std::sync::mpsc::Sender<StreamChunk>,
    stop: Arc<AtomicBool>,
) {
    let stopped = || async {
        while !stop.load(Ordering::Acquire) {
            tokio::time::sleep(STOP_POLL).await;
        }
    };
    loop {
        let next = tokio::select! {
            next = audio.next_chunk() => next,
            _ = stopped() => return,
        };
        let (chunk, last) = match next {
            Ok(Some(data)) => (StreamChunk::Data(data), false),
            Ok(None) => (StreamChunk::End, true),
            Err(e) => (StreamChunk::Error(e.to_string()), true),
        };
        if tx.send(chunk).is_err() || last {
            return;
        }
    }
}
