//! Experimental generative request/response types (ALR-GEN-001 / PT-GEN-001).
//!
//! 中文：图像生成 / STT / TTS 的 Experimental 请求类型（无 HTTP driver；见 ALR-GEN-002）。
//!
//! Types only — no transport, no vendor path strings. Capability checks live on
//! [`crate::protocol::ProtocolManifest::supports_generative_for_model`].

use serde::{Deserialize, Serialize};

/// Experimental text-to-image request (capability: `image_generation`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

impl ImageGenerationRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            size: None,
            n: None,
            response_format: None,
        }
    }

    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }
}

/// Experimental image generation result (bytes or URL — drivers fill later).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageGenerationResult {
    pub model: String,
    #[serde(default)]
    pub images: Vec<GeneratedImage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeneratedImage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
}

/// Experimental speech-to-text request (capability: `speech_to_text`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechToTextRequest {
    pub model: String,
    /// Local path or opaque URI; drivers interpret (ALR-GEN-002).
    pub audio_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

impl SpeechToTextRequest {
    pub fn new(model: impl Into<String>, audio_source: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            audio_source: audio_source.into(),
            language: None,
            prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechToTextResult {
    pub model: String,
    pub text: String,
}

/// Experimental text-to-speech request (capability: `text_to_speech`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextToSpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

impl TextToSpeechRequest {
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            voice: None,
            response_format: None,
        }
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextToSpeechResult {
    pub model: String,
    /// Audio payload placeholder until ALR-GEN-002 drivers fill bytes.
    #[serde(default)]
    pub audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_gen_request_builders() {
        let req = ImageGenerationRequest::new("gpt-image-1", "a cat").with_size("1024x1024");
        assert_eq!(req.model, "gpt-image-1");
        assert_eq!(req.size.as_deref(), Some("1024x1024"));
    }

    #[test]
    fn stt_tts_constructors() {
        let stt = SpeechToTextRequest::new("whisper-1", "clip.wav");
        assert_eq!(stt.audio_source, "clip.wav");
        let tts = TextToSpeechRequest::new("tts-1", "hello").with_voice("alloy");
        assert_eq!(tts.voice.as_deref(), Some("alloy"));
    }
}
