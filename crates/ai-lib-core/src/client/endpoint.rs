//! Endpoint resolution and service calls

use crate::protocol::{EndpointConfig, ProtocolError, ServiceConfig};
use crate::{Error, Result};
use std::collections::HashMap;
use std::future::Future;

use super::core::AiClient;

pub trait EndpointExt {
    fn resolve_endpoint(&self, name: &str) -> Result<&EndpointConfig>;

    /// Call a generic service by name. The returned future is `Send` and safe to use across threads.
    fn call_service(
        &self,
        service_name: &str,
    ) -> impl Future<Output = Result<serde_json::Value>> + Send;

    /// List models available from the provider. The returned future is `Send` and safe to use across threads.
    fn list_remote_models(&self) -> impl Future<Output = Result<Vec<String>>> + Send;
}

/// Resolve an operation name against a provider `endpoints` map.
///
/// Falls back from `chat` → `chat_openai` for dual-API manifests that omit the
/// canonical `chat` key used by `AiClient` chat requests.
pub(crate) fn lookup_endpoint<'a>(
    endpoints: Option<&'a HashMap<String, EndpointConfig>>,
    name: &str,
) -> Option<&'a EndpointConfig> {
    let endpoints = endpoints?;
    if let Some(ep) = endpoints.get(name) {
        return Some(ep);
    }
    if name == "chat" {
        return endpoints.get("chat_openai");
    }
    None
}

impl EndpointExt for AiClient {
    fn resolve_endpoint(&self, name: &str) -> Result<&EndpointConfig> {
        lookup_endpoint(self.manifest.endpoints.as_ref(), name).ok_or_else(|| {
            Error::Protocol(ProtocolError::NotFound {
                id: name.to_string(),
                hint: Some(
                    "Expected endpoints.<name> in the provider manifest (common keys: chat, chat_openai)"
                        .to_string(),
                ),
            })
        })
    }

    /// Call a generic service by name.
    async fn call_service(&self, service_name: &str) -> Result<serde_json::Value> {
        let service = self
            .manifest
            .services
            .as_ref()
            .and_then(|services: &HashMap<String, ServiceConfig>| services.get(service_name))
            .ok_or_else(|| {
                Error::Protocol(ProtocolError::NotFound {
                    id: service_name.to_string(),
                    hint: None,
                })
            })?;

        self.transport
            .execute_service(
                &service.path,
                &service.method,
                service.headers.as_ref(),
                service.query_params.as_ref(),
            )
            .await
    }

    /// List models available from the provider.
    async fn list_remote_models(&self) -> Result<Vec<String>> {
        let response = self.call_service("list_models").await?;

        let models: Vec<String> = if let Some(data) = response.get("data") {
            data.as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|m| {
                    m.get("id")
                        .and_then(|id| id.as_str().map(|s| s.to_string()))
                })
                .collect()
        } else if let Some(models) = response.get("models") {
            // Gemini style
            models
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|m| {
                    m.get("name")
                        .and_then(|n| n.as_str().map(|s| s.to_string()))
                })
                .collect()
        } else {
            vec![]
        };

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(path: &str) -> EndpointConfig {
        EndpointConfig {
            path: path.to_string(),
            method: "POST".to_string(),
            adapter: Some("openai".to_string()),
        }
    }

    #[test]
    fn lookup_prefers_exact_chat_key() {
        let mut map = HashMap::new();
        map.insert("chat".to_string(), ep("/chat/completions"));
        map.insert("chat_openai".to_string(), ep("/alt"));
        let got = lookup_endpoint(Some(&map), "chat").expect("chat");
        assert_eq!(got.path, "/chat/completions");
    }

    #[test]
    fn lookup_falls_back_chat_to_chat_openai() {
        let mut map = HashMap::new();
        map.insert("chat_openai".to_string(), ep("/chat/completions"));
        map.insert("chat_anthropic".to_string(), ep("/anthropic/v1/messages"));
        let got = lookup_endpoint(Some(&map), "chat").expect("chat via alias");
        assert_eq!(got.path, "/chat/completions");
    }

    #[test]
    fn lookup_missing_returns_none() {
        let mut map = HashMap::new();
        map.insert("embeddings".to_string(), ep("/embeddings"));
        assert!(lookup_endpoint(Some(&map), "chat").is_none());
    }
}
