//! Groq text-to-speech backend.
//!
//! Groq's `/openai/v1/audio/speech` endpoint speaks the OpenAI request shape,
//! so a Groq backend is just an [`OpenAiCompatTts`] pinned to the Groq base
//! URL. The base URL is shared with [`super::create_backend`]; this module
//! also offers a named constructor for clarity and tests.

use super::openai_compat::OpenAiCompatTts;

/// Groq API endpoint for text-to-speech.
pub const GROQ_SPEECH_URL: &str = "https://api.groq.com/openai/v1/audio/speech";

/// Build a Groq TTS backend (an [`OpenAiCompatTts`] pinned to Groq). Groq
/// always requires an API key.
pub fn groq_backend(
    api_key: String,
    model: String,
    voice: String,
    response_format: String,
) -> OpenAiCompatTts {
    OpenAiCompatTts::new(
        GROQ_SPEECH_URL.to_string(),
        Some(api_key),
        model,
        voice,
        response_format,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::TtsBackend;

    #[tokio::test]
    async fn synthesize_rejects_empty_text() {
        let backend = groq_backend(
            "test-key".to_string(),
            "canopylabs/orpheus-v1-english".to_string(),
            "autumn".to_string(),
            "wav".to_string(),
        );
        let err = backend
            .synthesize("   ")
            .await
            .err()
            .expect("empty text must fail");
        assert!(err.to_string().contains("empty text"));
    }
}
