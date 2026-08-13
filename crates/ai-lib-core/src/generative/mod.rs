//! Experimental generative L-Exec drivers (ALR-GEN-002 / PT-GEN-002/003).
//!
//! 中文：按 manifest `endpoints.<key>` 解析 URL，经统一 HttpTransport 调用；
//! 能力门控用 `supports_generative_for_model`（omit≠false）。
//!
//! - OpenAI adapter: Images / STT / TTS shapes
//! - DashScope adapter: native multimodal-generation body for image_generation
//!
//! Does not replace legacy `stt` / `tts` OpenAI-hardcoded clients.

mod audio;
mod image;

pub use audio::{SpeechToTextClient, TextToSpeechClient};
pub use image::ImageGenerationClient;

use crate::protocol::{EndpointConfig, ProtocolManifest};
use crate::{Error, Result};

pub const KEY_IMAGE_GENERATION: &str = "image_generation";
pub const KEY_SPEECH_TO_TEXT: &str = "speech_to_text";
pub const KEY_TEXT_TO_SPEECH: &str = "text_to_speech";

/// Resolve `endpoints.<key>` from the manifest (required for generative ops).
pub fn resolve_generative_endpoint<'a>(
    manifest: &'a ProtocolManifest,
    key: &str,
) -> Result<&'a EndpointConfig> {
    let eps = manifest.endpoints.as_ref().ok_or_else(|| {
        Error::configuration(
            "manifest endpoints map missing; cannot resolve generative L-Exec path".to_string(),
        )
    })?;
    eps.get(key).ok_or_else(|| {
        Error::configuration(format!(
            "manifest endpoints.{key} missing; declare PT-GEN-002 L-Exec map"
        ))
    })
}

/// Gate + resolve: capability must be known-true for the model; endpoint must exist.
pub fn require_generative_endpoint<'a>(
    manifest: &'a ProtocolManifest,
    model: &str,
    key: &str,
) -> Result<&'a EndpointConfig> {
    if !manifest.supports_generative_for_model(model, key) {
        return Err(Error::configuration(format!(
            "model `{model}` does not declare model_capabilities.{key}=true (omit≠false fail-closed)"
        )));
    }
    resolve_generative_endpoint(manifest, key)
}

pub(crate) fn adapter_name(ep: &EndpointConfig) -> &str {
    ep.adapter.as_deref().unwrap_or("openai")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::join_base_and_path;

    #[test]
    fn join_absolute_path_ignores_base() {
        let abs =
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";
        assert_eq!(
            join_base_and_path("https://dashscope.aliyuncs.com/compatible-mode/v1", abs),
            abs
        );
    }

    #[test]
    fn join_relative_path_prefixes_base() {
        assert_eq!(
            join_base_and_path("https://api.openai.com/v1", "/images/generations"),
            "https://api.openai.com/v1/images/generations"
        );
    }

    #[test]
    fn resolve_openai_and_qwen_image_endpoints() {
        let openai: ProtocolManifest = serde_yaml::from_str(
            r#"
id: openai
protocol_version: "2.0"
status: stable
category: ai_provider
official_url: "https://example.com"
support_contact: "s"
endpoint:
  base_url: "https://api.openai.com/v1"
capabilities:
  required: [text]
  optional: [image_generation]
endpoints:
  image_generation:
    path: /images/generations
    method: POST
    adapter: openai
metadata:
  models:
    gpt-image-1:
      model_capabilities:
        image_generation: true
"#,
        )
        .expect("openai");
        let ep =
            require_generative_endpoint(&openai, "gpt-image-1", KEY_IMAGE_GENERATION).expect("ep");
        assert_eq!(ep.path, "/images/generations");
        assert_eq!(adapter_name(ep), "openai");

        let qwen: ProtocolManifest = serde_yaml::from_str(
            r#"
id: qwen
protocol_version: "2.0"
status: stable
category: ai_provider
official_url: "https://example.com"
support_contact: "s"
endpoint:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
capabilities:
  required: [text]
  optional: [image_generation]
endpoints:
  image_generation:
    path: https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
    method: POST
    adapter: dashscope
metadata:
  models:
    qwen-image-plus:
      model_capabilities:
        image_generation: true
"#,
        )
        .expect("qwen");
        let ep = require_generative_endpoint(&qwen, "qwen-image-plus", KEY_IMAGE_GENERATION)
            .expect("ep");
        assert!(ep.path.starts_with("https://"));
        assert_eq!(adapter_name(ep), "dashscope");
        assert!(require_generative_endpoint(&qwen, "missing", KEY_IMAGE_GENERATION).is_err());
    }
}
