//! Helpers for ancillary clients (embeddings / stt / tts / rerank) to share
//! [`HttpTransport`] with chat ([GOV-007]).
//!
//! 附属 API 共用 HttpTransport，避免独立 reqwest 客户端。

use crate::protocol::ProtocolManifest;
use crate::transport::HttpTransport;
use crate::Result;

/// Minimal bearer manifest used when callers supply base_url + api_key without a
/// full protocol document. Credential is always applied via `credential_override`.
pub fn synthetic_bearer_manifest(base_url: &str) -> ProtocolManifest {
    serde_json::from_value(serde_json::json!({
        "id": "ancillary",
        "protocol_version": "1.0",
        "endpoint": {
            "base_url": base_url,
            "auth": { "type": "bearer", "token_env": "AI_LIB_ANCILLARY_API_KEY" }
        },
        "capabilities": { "streaming": false, "tools": false, "vision": false },
        "provider_id": "ancillary",
        "status": "stable",
        "category": "ai_provider",
        "official_url": "",
        "support_contact": ""
    }))
    .expect("synthetic ancillary manifest")
}

/// Build [`HttpTransport`] for ancillary APIs, preferring a real manifest when present.
pub fn build_ancillary_transport(
    manifest: Option<&ProtocolManifest>,
    base_url: &str,
    model: &str,
    api_key: &str,
) -> Result<HttpTransport> {
    match manifest {
        Some(m) => HttpTransport::new_with_base_url_and_credential(
            m,
            model,
            Some(base_url),
            Some(api_key),
        ),
        None => {
            let synthetic = synthetic_bearer_manifest(base_url);
            HttpTransport::new_with_base_url_and_credential(
                &synthetic,
                model,
                Some(base_url),
                Some(api_key),
            )
        }
    }
}

/// Ensure path begins with `/` for HttpTransport URL concatenation.
pub fn normalize_endpoint_path(path: String) -> String {
    if path.starts_with('/') {
        path
    } else {
        format!("/{path}")
    }
}
