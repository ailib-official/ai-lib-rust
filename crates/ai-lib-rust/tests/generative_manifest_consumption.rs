//! Generative manifest consumption tests.
//!
//! Verifies ai-lib-rust can parse and utilize latest ai-protocol V2 provider
//! manifests for generative/multimodal capabilities.
#![cfg(feature = "multimodal")]

use ai_lib_rust::multimodal::{Modality, MultimodalCapabilities};
use ai_lib_rust::protocol::v2::manifest::{ApiStyle, ManifestV2};
use std::fs;
use std::path::PathBuf;

fn resolve_ai_protocol_root() -> PathBuf {
    if let Ok(path) = std::env::var("AI_PROTOCOL_DIR") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("AI_PROTOCOL_PATH") {
        return PathBuf::from(path);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../ai-protocol"),
        manifest_dir.join("../../ai-protocol"),
        PathBuf::from("d:/ai-protocol"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("Unable to locate ai-protocol root for manifest consumption test");
}

#[test]
fn consume_latest_v2_generative_manifests() {
    let root = resolve_ai_protocol_root();
    // CI PROTO-PIN is often ai-protocol@v1.0.0 (pre-PT-ARCH-005: google.yaml).
    // Post-#31 tip uses gemini.yaml + aliases:[google]. Resolve either.
    let providers: &[(&str, &[&str])] = &[
        ("gemini_family", &["gemini", "google"]),
        ("deepseek", &["deepseek"]),
        ("qwen", &["qwen"]),
        ("doubao", &["doubao"]),
    ];

    for (label, candidates) in providers {
        let (stem, path, raw) = candidates
            .iter()
            .map(|id| {
                let path = root.join(format!("v2/providers/{id}.yaml"));
                (id, path.clone(), fs::read_to_string(&path).ok())
            })
            .find_map(|(id, path, raw)| raw.map(|r| (*id, path, r)))
            .unwrap_or_else(|| {
                panic!(
                    "failed reading v2 generative manifest for {label}; tried {:?} under {}",
                    candidates,
                    root.join("v2/providers").display()
                )
            });

        let manifest: ManifestV2 = serde_yaml::from_str(&raw).unwrap_or_else(|e| {
            panic!("failed parsing {}: {e}", path.display());
        });

        assert!(manifest.is_v2(), "{label} should be parsed as V2");
        assert_eq!(
            manifest.id, stem,
            "{label}: file stem {stem} should match manifest.id"
        );

        if *label == "gemini_family" {
            assert_eq!(manifest.detect_api_style(), ApiStyle::GeminiGenerate);
            // Tip protocol: canonical gemini + alias google. Legacy pin: id google, no aliases.
            if manifest.id == "gemini" {
                let aliases = manifest
                    .aliases
                    .as_ref()
                    .expect("gemini aliases on tip protocol");
                assert!(
                    aliases.iter().any(|a| a == "google"),
                    "gemini must declare google alias (PT-ARCH-005)"
                );
            }
        } else {
            assert_eq!(manifest.detect_api_style(), ApiStyle::OpenAiCompatible);
        }

        let multimodal = manifest
            .multimodal
            .as_ref()
            .expect("multimodal section required");
        let caps = MultimodalCapabilities::from_config(multimodal);

        assert!(caps.supports_input(Modality::Text));
        assert!(caps.supports_output(Modality::Text));
        if *label == "qwen" || *label == "gemini_family" {
            assert!(
                caps.supports_input(Modality::Video),
                "{label} should support video input"
            );
        }

        // Latest schema includes output.video declaration; runtimes must not drop it.
        let output_video_supported = multimodal
            .output
            .as_ref()
            .and_then(|o| o.video.as_ref())
            .map(|v| v.supported)
            .unwrap_or(false);
        assert!(
            !output_video_supported,
            "{label} output.video expected false in current P0 manifests"
        );
    }
}

#[test]
fn supports_structured_endpoint_chat_shape() {
    let raw = r#"
id: shape-compat
protocol_version: "2.0"
endpoint:
  base_url: "https://example.com"
  chat:
    path: "/v2/chat"
    method: "POST"
capabilities:
  required: ["text"]
  optional: []
"#;
    let manifest: ManifestV2 = serde_yaml::from_str(raw).expect("manifest should parse");
    assert_eq!(manifest.chat_path(), "/v2/chat");
    assert_eq!(manifest.base_url(), "https://example.com");
}

#[test]
fn consume_wave1_v2_provider_manifests() {
    let root = resolve_ai_protocol_root();
    let providers = ["cohere", "moonshot", "zhipu", "jina"];

    for provider in providers {
        let path = root.join(format!("v2/providers/{provider}.yaml"));
        let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("failed reading {}: {e}", path.display());
        });
        let manifest: ManifestV2 = serde_yaml::from_str(&raw).unwrap_or_else(|e| {
            panic!("failed parsing {}: {e}", path.display());
        });

        assert!(manifest.is_v2(), "{provider} should be parsed as V2");
        assert_eq!(manifest.id, provider);
        assert!(
            !manifest.base_url().is_empty(),
            "{provider} should expose base_url"
        );

        let chat_path = manifest.endpoint.chat.as_ref().map(|p| p.as_path());
        let rerank_path = manifest.endpoint.rerank.as_ref().map(|p| p.as_path());

        match provider {
            "cohere" => {
                assert_eq!(chat_path, Some("/chat"));
                assert_eq!(rerank_path, Some("/rerank"));
            }
            "moonshot" | "zhipu" => {
                assert_eq!(chat_path, Some("/chat/completions"));
            }
            "jina" => {
                assert!(chat_path.is_none(), "jina should not expose chat path");
                assert_eq!(rerank_path, Some("/v1/rerank"));
            }
            _ => unreachable!(),
        }

        if provider == "moonshot" {
            let multimodal = manifest
                .multimodal
                .as_ref()
                .expect("moonshot should expose multimodal section");
            let caps = MultimodalCapabilities::from_config(multimodal);
            assert!(
                caps.supports_input(Modality::Video),
                "moonshot should support video input"
            );
            let output_video_supported = multimodal
                .output
                .as_ref()
                .and_then(|o| o.video.as_ref())
                .map(|v| v.supported)
                .unwrap_or(false);
            assert!(
                !output_video_supported,
                "moonshot output.video should remain disabled in current contract"
            );
        }
    }
}
