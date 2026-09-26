//! Text-to-speech backends: trait definition and implementations.
//!
//! The synthesis stage of the "read selection aloud" feature lives behind a
//! small [`TtsBackend`] trait (mirroring [`crate::transcription::TranscriptionBackend`]).
//!
//! Cloud and local-server backends are covered today:
//! - `groq` / `openai` / `tts-sidecar` share the OpenAI `/v1/audio/speech`
//!   request shape via [`openai_compat::OpenAiCompatTts`]. The `tts-sidecar`
//!   backend points at any local OpenAI-compatible server (Kokoro, Supertonic,
//!   etc.) and needs no API key.
//! - `deepgram` uses Deepgram Aura-2 via [`deepgram_aura::DeepgramAuraTts`].
//!
//! Roadmap: native in-process local backends (Supertonic via ONNX/`ort`,
//! Piper, Kokoro) are future work. Until then, those models are reachable
//! today through a local server behind the `tts-sidecar` backend.

pub mod deepgram_aura;
pub mod groq;
pub mod openai_compat;

use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::{TtsConfig, WhisrsError};

/// Default endpoint for a local OpenAI-compatible TTS sidecar (e.g. Kokoro-FastAPI).
const DEFAULT_SIDECAR_URL: &str = "http://127.0.0.1:8880/v1/audio/speech";

/// OpenAI's text-to-speech endpoint.
const OPENAI_SPEECH_URL: &str = "https://api.openai.com/v1/audio/speech";

// Per-backend default model/voice, applied when `[tts] model`/`voice` are unset
// so a user can switch `backend` without also overriding the model. The Groq
// default (orpheus) is meaningless to OpenAI/Deepgram, which is why the default
// is resolved here per backend rather than baked into the config.
const GROQ_DEFAULT_MODEL: &str = "canopylabs/orpheus-v1-english";
const GROQ_DEFAULT_VOICE: &str = "autumn";
const OPENAI_DEFAULT_MODEL: &str = "gpt-4o-mini-tts";
const OPENAI_DEFAULT_VOICE: &str = "alloy";
const SIDECAR_DEFAULT_MODEL: &str = "kokoro";
const SIDECAR_DEFAULT_VOICE: &str = "af_heart";

/// Trait for text-to-speech backends.
///
/// Backends can return a whole WAV or send its body incrementally for playback.
#[async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize `text` into speech, returning WAV-encoded audio bytes.
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>, WhisrsError>;

    /// Send WAV body chunks as they arrive; returns when the response ends.
    async fn synthesize_stream(
        &self,
        text: &str,
        sender: mpsc::Sender<Vec<u8>>,
        stop: Arc<AtomicBool>,
    ) -> Result<(), WhisrsError>;
}

/// Forward an HTTP response to the bounded playback queue without buffering it.
pub(crate) async fn stream_response(
    mut response: reqwest::Response,
    sender: mpsc::Sender<Vec<u8>>,
    stop: Arc<AtomicBool>,
) -> Result<(), WhisrsError> {
    loop {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let chunk = tokio::select! {
            chunk = response.chunk() => chunk.map_err(|e| WhisrsError::Transcription(format!("TTS read body failed: {e}")))?,
            _ = sender.closed() => return Ok(()),
            _ = async {
                while !stop.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            } => return Ok(()),
        };
        let Some(chunk) = chunk else {
            return Ok(());
        };
        tokio::select! {
            result = sender.send(chunk.to_vec()) => { if result.is_err() { return Ok(()); } },
            _ = async {
                while !stop.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            } => return Ok(()),
        }
    }
}

/// Build the configured TTS backend.
///
/// `api_key` should already be resolved for the configured backend (see the
/// daemon's `resolve_tts_api_key`). Cloud backends (`groq`, `openai`,
/// `deepgram`) require a key; the `tts-sidecar` backend treats it as optional.
pub fn create_backend(
    config: &TtsConfig,
    api_key: Option<String>,
) -> Result<Box<dyn TtsBackend>, WhisrsError> {
    let api_key = api_key.filter(|k| !k.is_empty());

    // Require a key for the cloud backends; the sidecar can run keyless.
    let require_key = |key: Option<String>| -> Result<String, WhisrsError> {
        key.ok_or_else(|| {
            WhisrsError::Config(
                "TTS is enabled but no API key is configured.\n\
                 Add an api_key to [tts], or configure the backend's key \
                 (e.g. [groq] api_key / WHISRS_GROQ_API_KEY)."
                    .to_string(),
            )
        })
    };

    // Resolve the effective model/voice: the configured value when present and
    // non-blank, otherwise the selected backend's default.
    let model_or = |default: &str| -> String {
        config
            .model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(default)
            .to_string()
    };
    let voice_or = |default: &str| -> String {
        config
            .voice
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(default)
            .to_string()
    };

    match config.backend.as_str() {
        "groq" => Ok(Box::new(openai_compat::OpenAiCompatTts::new(
            groq::GROQ_SPEECH_URL.to_string(),
            Some(require_key(api_key)?),
            model_or(GROQ_DEFAULT_MODEL),
            voice_or(GROQ_DEFAULT_VOICE),
            config.response_format.clone(),
        ))),
        "openai" => Ok(Box::new(openai_compat::OpenAiCompatTts::new(
            OPENAI_SPEECH_URL.to_string(),
            Some(require_key(api_key)?),
            model_or(OPENAI_DEFAULT_MODEL),
            voice_or(OPENAI_DEFAULT_VOICE),
            config.response_format.clone(),
        ))),
        "tts-sidecar" | "openai-compat" => {
            let base_url = config
                .url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(DEFAULT_SIDECAR_URL)
                .to_string();
            Ok(Box::new(openai_compat::OpenAiCompatTts::new(
                base_url,
                api_key, // optional — sidecars usually need none
                model_or(SIDECAR_DEFAULT_MODEL),
                voice_or(SIDECAR_DEFAULT_VOICE),
                config.response_format.clone(),
            )))
        }
        // `[tts] voice` is intentionally not passed here: Aura encodes the
        // voice in the model id (e.g. `aura-2-thalia-en`), so the model alone
        // selects the voice.
        "deepgram" => Ok(Box::new(deepgram_aura::DeepgramAuraTts::new(
            require_key(api_key)?,
            model_or(deepgram_aura::DEFAULT_MODEL),
        ))),
        other => Err(WhisrsError::Config(format!(
            "Unknown TTS backend '{other}'. Valid options: groq, openai, tts-sidecar, deepgram"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(backend: &str) -> TtsConfig {
        TtsConfig {
            enabled: true,
            backend: backend.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn create_groq_backend_requires_key() {
        assert!(create_backend(&cfg("groq"), Some("k".to_string())).is_ok());
        assert!(create_backend(&cfg("groq"), None).is_err());
    }

    #[test]
    fn create_openai_backend_requires_key() {
        assert!(create_backend(&cfg("openai"), Some("k".to_string())).is_ok());
        assert!(create_backend(&cfg("openai"), None).is_err());
    }

    #[test]
    fn create_deepgram_backend_requires_key() {
        assert!(create_backend(&cfg("deepgram"), Some("k".to_string())).is_ok());
        assert!(create_backend(&cfg("deepgram"), None).is_err());
    }

    #[test]
    fn create_sidecar_backend_allows_no_key() {
        // Sidecar must build with no key; alias must also work.
        assert!(create_backend(&cfg("tts-sidecar"), None).is_ok());
        assert!(create_backend(&cfg("openai-compat"), None).is_ok());
    }

    #[test]
    fn create_unknown_backend_errors() {
        // `Box<dyn TtsBackend>` isn't Debug, so match instead of unwrap_err().
        match create_backend(&cfg("bogus"), Some("k".to_string())) {
            Err(e) => assert!(e.to_string().contains("Unknown TTS backend")),
            Ok(_) => panic!("expected an error for an unknown backend"),
        }
    }
}
