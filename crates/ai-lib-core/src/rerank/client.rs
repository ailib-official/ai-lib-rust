//! Rerank client for document relevance scoring.
//!
//! Protocol-driven construction: prefer [`RerankerClientBuilder::from_manifest`] /
//! [`RerankerClientBuilder::from_model`]. No silent Cohere host default ([ARCH-001]).

use super::types::{RerankOptions, RerankResult};
use crate::credentials::{self, resolve_credential};
use crate::protocol::{ProtocolLoader, ProtocolManifest};
use crate::{Error, ErrorContext, Result};

/// Client for document reranking.
pub struct RerankerClient {
    http_client: reqwest::Client,
    model: String,
    base_url: String,
    endpoint_path: String,
    api_key: String,
}

impl RerankerClient {
    pub fn builder() -> RerankerClientBuilder {
        RerankerClientBuilder::new()
    }

    pub async fn rerank(
        &self,
        query: &str,
        documents: &[impl AsRef<str>],
        options: &RerankOptions,
    ) -> Result<Vec<RerankResult>> {
        let endpoint = join_url(&self.base_url, &self.endpoint_path);
        let docs: Vec<String> = documents.iter().map(|d| d.as_ref().to_string()).collect();
        let mut body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": docs,
        });
        if let Some(top_n) = options.top_n {
            body["top_n"] = serde_json::json!(top_n);
        }
        if let Some(max_tokens) = options.max_tokens_per_doc {
            body["max_tokens_per_doc"] = serde_json::json!(max_tokens);
        }
        let response = self
            .http_client
            .post(&endpoint)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("Rerank request failed: {}", e),
                    ErrorContext::new().with_source("rerank"),
                )
            })?;
        let status = response.status();
        let body_str = response.text().await.map_err(|e| {
            Error::network_with_context(
                format!("Failed to read Rerank response: {}", e),
                ErrorContext::new(),
            )
        })?;
        if !status.is_success() {
            return Err(Error::api_with_context(
                format!("Rerank API error ({}): {}", status, body_str),
                ErrorContext::new(),
            ));
        }
        let json: serde_json::Value = serde_json::from_str(&body_str)?;
        let results = json
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                Error::api_with_context(
                    "Invalid rerank response: missing results",
                    ErrorContext::new(),
                )
            })?;
        let mut out = Vec::with_capacity(results.len());
        for r in results {
            let index = r.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let relevance_score = r
                .get("relevance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let document = r.get("document").and_then(|v| v.as_str()).map(String::from);
            out.push(RerankResult {
                index,
                relevance_score,
                document,
            });
        }
        Ok(out)
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{}", path)
    };
    format!("{}{}", base, path)
}

fn rerank_path_from_manifest(manifest: &ProtocolManifest) -> String {
    if let Some(eps) = &manifest.endpoints {
        if let Some(ep) = eps.get("rerank") {
            return ep.path.clone();
        }
    }
    if let Some(services) = &manifest.services {
        if let Some(svc) = services.get("rerank") {
            return svc.path.clone();
        }
    }
    "/rerank".to_string()
}

pub struct RerankerClientBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    endpoint_path: Option<String>,
    timeout_secs: u64,
    protocol_path: Option<String>,
}

impl RerankerClientBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            endpoint_path: None,
            timeout_secs: 60,
            protocol_path: None,
        }
    }
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    pub fn endpoint_path(mut self, path: impl Into<String>) -> Self {
        self.endpoint_path = Some(path.into());
        self
    }
    pub fn protocol_path(mut self, path: impl Into<String>) -> Self {
        self.protocol_path = Some(path.into());
        self
    }

    pub fn from_manifest(
        mut self,
        manifest: &ProtocolManifest,
        model_id: impl Into<String>,
    ) -> Result<Self> {
        let cred = resolve_credential(manifest, self.api_key.as_deref());
        let secret = cred.secret().ok_or_else(|| {
            Error::configuration(format!(
                "API key required for rerank (provider={}; tried {:?})",
                credentials::provider_id(manifest),
                cred.required_envs
                    .iter()
                    .chain(cred.conventional_envs.iter())
                    .cloned()
                    .collect::<Vec<_>>()
            ))
        })?;
        self.api_key = Some(secret.to_string());
        self.base_url = Some(manifest.get_base_url().to_string());
        if self.endpoint_path.is_none() {
            self.endpoint_path = Some(rerank_path_from_manifest(manifest));
        }
        self.model = Some(model_id.into());
        Ok(self)
    }

    pub async fn from_model(self, model: &str) -> Result<RerankerClient> {
        let mut loader = ProtocolLoader::new();
        if let Some(path) = &self.protocol_path {
            loader = loader.with_base_path(path);
        }
        let manifest = loader.load_model(model).await.map_err(Error::Protocol)?;
        let parts: Vec<&str> = model.split('/').collect();
        let model_id = if parts.len() >= 2 {
            parts[1..].join("/")
        } else {
            model.to_string()
        };
        self.from_manifest(&manifest, model_id)?.build().await
    }

    pub async fn build(self) -> Result<RerankerClient> {
        let model = self
            .model
            .ok_or_else(|| Error::configuration("Model must be specified"))?;
        let api_key = self.api_key.ok_or_else(|| {
            Error::configuration(
                "API key required: use from_manifest/from_model or set api_key explicitly",
            )
        })?;
        let base_url = self.base_url.ok_or_else(|| {
            Error::configuration(
                "base_url required: use from_manifest/from_model or set base_url explicitly (no vendor default)",
            )
        })?;
        let endpoint_path = self.endpoint_path.unwrap_or_else(|| "/rerank".to_string());
        let endpoint_path = if endpoint_path.starts_with('/') {
            endpoint_path
        } else {
            format!("/{}", endpoint_path)
        };
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| Error::configuration(format!("Failed to create HTTP client: {}", e)))?;
        Ok(RerankerClient {
            http_client,
            model,
            base_url,
            endpoint_path,
            api_key,
        })
    }
}

impl Default for RerankerClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn build_without_base_url_errors() {
        let res = RerankerClient::builder()
            .model("rerank-v3")
            .api_key("k")
            .build()
            .await;
        assert!(res.is_err());
        assert!(res.err().unwrap().to_string().contains("base_url"));
    }
}
