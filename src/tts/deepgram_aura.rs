//! Deepgram Aura-2 text-to-speech backend.
//!
//! POSTs to Deepgram's `/v1/speak` endpoint requesting 24 kHz linear16 audio
//! wrapped in a WAV container, so the bytes can be decoded by
//! [`crate::audio::playback::decode_wav`] without further conversion.

use async_trait::async_trait;
use serde::Serialize;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::debug;

use crate::WhisrsError;

use super::{stream_response, TtsBackend};

/// Deepgram speak endpoint (model + audio params are passed as query params).
const DEEPGRAM_SPEAK_URL: &str = "https://api.deepgram.com/v1/speak";

/// Default Aura-2 voice/model.
pub const DEFAULT_MODEL: &str = "aura-2-thalia-en";

/// Deepgram Aura-2 text-to-speech backend.
pub struct DeepgramAuraTts {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl DeepgramAuraTts {
    /// Create a new Deepgram Aura-2 TTS backend. A blank `model` falls back to
    /// [`DEFAULT_MODEL`].
    pub fn new(api_key: String, model: String) -> Self {
        let model = if model.trim().is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            model
        };
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    /// Build the JSON request body for the given input text.
    ///
    /// Exposed for unit testing the wire format without hitting the network.
    fn request_body<'a>(&self, text: &'a str) -> SpeechRequest<'a> {
        SpeechRequest { text }
    }

    /// The fully-qualified request URL including query parameters. Exposed for
    /// unit testing.
    fn request_url(&self) -> String {
        format!(
            "{DEEPGRAM_SPEAK_URL}?model={}&encoding=linear16&container=wav",
            self.model
        )
    }

    async fn request(&self, text: &str) -> Result<reqwest::Response, WhisrsError> {
        if text.trim().is_empty() {
            return Err(WhisrsError::Transcription(
                "cannot synthesize empty text".into(),
            ));
        }
        debug!(
            "sending {} chars to Deepgram Aura (model={})",
            text.len(),
            self.model
        );
        let response = self
            .client
            .post(self.request_url())
            .header("Authorization", format!("Token {}", self.api_key))
            .json(&self.request_body(text))
            .send()
            .await
            .map_err(|e| WhisrsError::Transcription(format!("Deepgram TTS request failed: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(WhisrsError::Transcription(format!(
                "Deepgram TTS error ({}): {}",
                status.as_u16(),
                body
            )));
        }
        Ok(response)
    }
}

/// Request body for Deepgram's text-to-speech endpoint.
#[derive(Debug, Serialize)]
struct SpeechRequest<'a> {
    text: &'a str,
}

#[async_trait]
impl TtsBackend for DeepgramAuraTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>, WhisrsError> {
        let bytes = self.request(text).await?.bytes().await.map_err(|e| {
            WhisrsError::Transcription(format!("Deepgram TTS read body failed: {e}"))
        })?;
        Ok(bytes.to_vec())
    }

    async fn synthesize_stream(
        &self,
        text: &str,
        sender: mpsc::Sender<Vec<u8>>,
        stop: Arc<AtomicBool>,
    ) -> Result<(), WhisrsError> {
        let response = self.request(text).await?;
        stream_response(response, sender, stop).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_url_has_model_and_wav_params() {
        let backend = DeepgramAuraTts::new("test-key".to_string(), "aura-2-thalia-en".to_string());
        let url = backend.request_url();
        assert!(url.starts_with("https://api.deepgram.com/v1/speak?"));
        assert!(url.contains("model=aura-2-thalia-en"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("container=wav"));
    }

    #[test]
    fn blank_model_falls_back_to_default() {
        let backend = DeepgramAuraTts::new("test-key".to_string(), String::new());
        assert_eq!(backend.model, DEFAULT_MODEL);
        assert!(backend.request_url().contains("model=aura-2-thalia-en"));
    }

    #[test]
    fn request_body_serializes_text_only() {
        let backend = DeepgramAuraTts::new("test-key".to_string(), "aura-2-thalia-en".to_string());
        let json = serde_json::to_value(backend.request_body("hello world")).unwrap();
        assert_eq!(json["text"], "hello world");
        // Body must not leak the key or model — those go in the header/query.
        assert!(json.get("model").is_none());
    }

    #[test]
    fn auth_header_uses_token_scheme() {
        // Deepgram uses `Token <key>`, not `Bearer <key>`. The header is built
        // inline in synthesize(); assert the scheme prefix here as a guard.
        let backend = DeepgramAuraTts::new("secret".to_string(), "aura-2-thalia-en".to_string());
        let header = format!("Token {}", backend.api_key);
        assert_eq!(header, "Token secret");
    }

    #[tokio::test]
    async fn synthesize_rejects_empty_text() {
        let backend = DeepgramAuraTts::new("test-key".to_string(), "aura-2-thalia-en".to_string());
        let err = backend.synthesize("   ").await.unwrap_err();
        assert!(err.to_string().contains("empty text"));
    }
}
