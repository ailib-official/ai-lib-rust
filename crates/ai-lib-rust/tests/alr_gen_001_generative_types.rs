//! ALR-GEN-001 — Experimental generative capability checks (omit ≠ false).

use ai_lib_rust::protocol::ProtocolManifest;
use ai_lib_rust::types::{ImageGenerationRequest, SpeechToTextRequest, TextToSpeechRequest};

#[test]
fn omit_generative_keys_do_not_pass_as_false_fact() {
    let manifest: ProtocolManifest = serde_yaml::from_str(
        r#"
id: gen-omit
protocol_version: "2.0"
status: stable
category: ai_provider
official_url: "https://example.com"
support_contact: "s"
endpoint:
  base_url: "https://api.example.com"
capabilities:
  required: [text]
  optional: []
metadata:
  models:
    chat-only:
      context_window: 128000
      model_capabilities:
        tool_call: true
"#,
    )
    .expect("manifest");

    // Omitted generative keys → Unknown → fail-closed false (not a false *fact*).
    assert!(!manifest.supports_generative_for_model("chat-only", "image_generation"));
    assert!(!manifest.supports_generative_for_model("chat-only", "speech_to_text"));
    assert!(!manifest.supports_generative_for_model("missing-model", "image_generation"));

    let entry = manifest
        .metadata_model_entry("chat-only")
        .expect("entry");
    use ai_lib_rust::protocol::CapabilityKnown;
    assert_eq!(
        entry.supports_generative_capability("image_generation"),
        CapabilityKnown::Unknown
    );
}

#[test]
fn image_generation_present_passes_check() {
    let manifest: ProtocolManifest = serde_yaml::from_str(
        r#"
id: gen-img
protocol_version: "2.0"
status: stable
category: ai_provider
official_url: "https://example.com"
support_contact: "s"
endpoint:
  base_url: "https://api.example.com"
capabilities:
  required: [text]
  optional: [image_generation]
metadata:
  models:
    gpt-image-1:
      model_capabilities:
        image_generation: true
      modalities:
        input: [text]
        output: [image]
"#,
    )
    .expect("manifest");

    assert!(manifest.supports_generative_for_model("gpt-image-1", "image_generation"));
    assert!(!manifest.supports_generative_for_model("gpt-image-1", "speech_to_text"));
}

#[test]
fn experimental_request_types_construct() {
    let _ = ImageGenerationRequest::new("gpt-image-1", "draw");
    let _ = SpeechToTextRequest::new("whisper-1", "a.wav");
    let _ = TextToSpeechRequest::new("tts-1", "hi");
}
