//! Embedding client for generating embeddings.
//!
//! Protocol-driven construction: prefer [`EmbeddingClientBuilder::from_manifest`] /
//! [`EmbeddingClientBuilder::from_model`]. Base URL and credentials come from the
//! provider manifest ([ARCH-001]); there is no silent default to a vendor host.

use super::types::{Embedding, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};
use crate::credentials::{self, resolve_credential};
use crate::protocol::{ProtocolLoader, ProtocolManifest};
use crate::{Error, ErrorContext, Result};

pub struct EmbeddingClient {
    http_client: reqwest::Client,
    model: String,
    base_url: String,
    endpoint_path: String,
    api_key: String,
    dimensions: Option<usize>,
    max_batch_size: usize,
}

impl EmbeddingClient {
    pub fn builder() -> EmbeddingClientBuilder {
        EmbeddingClientBuilder::new()
    }

    pub async fn embed(&self, text: &str) -> Result<EmbeddingResponse> {
        let request = EmbeddingRequest::single(&self.model, text);
        self.execute(request).await
    }

    pub async fn embed_batch(&self, texts: &[impl AsRef<str>]) -> Result<EmbeddingResponse> {
        let texts: Vec<String> = texts.iter().map(|t| t.as_ref().to_string()).collect();
        if texts.len() <= self.max_batch_size {
            return self
                .execute(EmbeddingRequest::batch(&self.model, texts))
                .await;
        }
        let mut all_embeddings: Vec<Embedding> = Vec::new();
        let mut total_usage = EmbeddingUsage::default();
        for (batch_idx, chunk) in texts.chunks(self.max_batch_size).enumerate() {
            let response = self
                .execute(EmbeddingRequest::batch(&self.model, chunk.to_vec()))
                .await?;
            let offset = batch_idx * self.max_batch_size;
            for mut emb in response.embeddings {
                emb.index += offset;
                all_embeddings.push(emb);
            }
            total_usage.add(&response.usage);
        }
        Ok(EmbeddingResponse::new(
            all_embeddings,
            self.model.clone(),
            total_usage,
        ))
    }

    async fn execute(&self, mut request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        if let Some(dims) = self.dimensions {
            request = request.with_dimensions(dims);
        }
        let endpoint = join_url(&self.base_url, &self.endpoint_path);
        let response = self
            .http_client
            .post(&endpoint)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("Embedding request failed: {}", e),
                    ErrorContext::new().with_source("embeddings"),
                )
            })?;
        let status = response.status();
        let body = response.text().await.map_err(|e| {
            Error::network_with_context(
                format!("Failed to read response: {}", e),
                ErrorContext::new(),
            )
        })?;
        if !status.is_success() {
            return Err(Error::api_with_context(
                format!("Embedding API error ({}): {}", status, body),
                ErrorContext::new(),
            ));
        }
        let json: serde_json::Value = serde_json::from_str(&body)?;
        EmbeddingResponse::from_openai_format(&json)
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

/// Resolve embeddings path from manifest endpoints / services, else OpenAI-compat `/embeddings`.
fn embeddings_path_from_manifest(manifest: &ProtocolManifest) -> String {
    if let Some(eps) = &manifest.endpoints {
        if let Some(ep) = eps.get("embeddings") {
            return ep.path.clone();
        }
    }
    if let Some(services) = &manifest.services {
        if let Some(svc) = services.get("embeddings") {
            return svc.path.clone();
        }
    }
    "/embeddings".to_string()
}

pub struct EmbeddingClientBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    endpoint_path: Option<String>,
    dimensions: Option<usize>,
    max_batch_size: usize,
    timeout_secs: u64,
    protocol_path: Option<String>,
}

impl EmbeddingClientBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            endpoint_path: None,
            dimensions: None,
            max_batch_size: 100,
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
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
    pub fn protocol_path(mut self, path: impl Into<String>) -> Self {
        self.protocol_path = Some(path.into());
        self
    }

    /// Build from an already-loaded protocol manifest ([ARCH-001]).
    pub fn from_manifest(
        mut self,
        manifest: &ProtocolManifest,
        model_id: impl Into<String>,
    ) -> Result<Self> {
        let cred = resolve_credential(manifest, self.api_key.as_deref());
        let secret = cred.secret().ok_or_else(|| {
            Error::configuration(format!(
                "API key required for embeddings (provider={}; tried {:?})",
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
            self.endpoint_path = Some(embeddings_path_from_manifest(manifest));
        }
        self.model = Some(model_id.into());
        Ok(self)
    }

    /// Load provider/model via [`ProtocolLoader`] then build.
    ///
    /// `model` uses `provider/model-id` form (same as [`crate::AiClientBuilder`]).
    pub async fn from_model(self, model: &str) -> Result<EmbeddingClient> {
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

    pub async fn build(self) -> Result<EmbeddingClient> {
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
        let endpoint_path = self
            .endpoint_path
            .unwrap_or_else(|| "/embeddings".to_string());
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| Error::configuration(format!("Failed to create HTTP client: {}", e)))?;
        Ok(EmbeddingClient {
            http_client,
            model,
            base_url,
            endpoint_path,
            api_key,
            dimensions: self.dimensions,
            max_batch_size: self.max_batch_size,
        })
    }
}

impl Default for EmbeddingClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_manifest(base: &str) -> ProtocolManifest {
        serde_json::from_value(serde_json::json!({
            "id": "testprov",
            "protocol_version": "1.0",
            "endpoint": {
                "base_url": base,
                "auth": { "type": "bearer", "token_env": "TESTPROV_API_KEY" }
            },
            "capabilities": { "streaming": false, "tools": false, "vision": false },
            "provider_id": "testprov",
            "status": "stable",
            "category": "ai_provider",
            "official_url": "",
            "support_contact": ""
        }))
        .expect("test manifest")
    }

    #[tokio::test]
    async fn from_manifest_requires_no_vendor_default() {
        std::env::set_var("TESTPROV_API_KEY", "sk-test");
        let m = minimal_manifest("https://example.test/v1");
        let client = EmbeddingClient::builder()
            .from_manifest(&m, "emb-1")
            .unwrap()
            .build()
            .await
            .unwrap();
        assert_eq!(client.model(), "emb-1");
        assert!(client.base_url.contains("example.test"));
        std::env::remove_var("TESTPROV_API_KEY");
    }

    #[tokio::test]
    async fn build_without_base_url_errors() {
        let res = EmbeddingClient::builder()
            .model("x")
            .api_key("k")
            .build()
            .await;
        assert!(res.is_err());
        assert!(res.err().unwrap().to_string().contains("base_url"));
    }
}
