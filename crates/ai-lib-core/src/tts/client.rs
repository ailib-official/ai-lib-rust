//! TTS (Text-to-Speech) client.
//! TTS（文本转语音）客户端。
//! HTTP via [`crate::transport::HttpTransport`] ([GOV-007]).

use super::types;
use super::types::{AudioOutput, TtsOptions};
use crate::protocol::ProtocolManifest;
use crate::transport::{build_ancillary_transport, normalize_endpoint_path, HttpTransport};
use crate::{Error, ErrorContext, Result};
use std::str::FromStr;

/// Client for text-to-speech synthesis.
pub struct TtsClient {
    transport: HttpTransport,
    model: String,
    endpoint_path: String,
}

impl TtsClient {
    pub fn builder() -> TtsClientBuilder {
        TtsClientBuilder::new()
    }

    pub async fn synthesize(&self, text: &str, options: &TtsOptions) -> Result<AudioOutput> {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": text,
        });
        if let Some(voice) = &options.voice {
            body["voice"] = serde_json::Value::String(voice.clone());
        }
        if let Some(speed) = options.speed {
            body["speed"] = serde_json::json!(speed);
        }
        if let Some(rf) = &options.response_format {
            body["response_format"] = serde_json::Value::String(rf.clone());
        }
        let (status, bytes) = self
            .transport
            .execute_bytes_post(&self.endpoint_path, &body)
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("TTS request failed: {e}"),
                    ErrorContext::new().with_source("tts"),
                )
            })?;
        if !status.is_success() {
            let body_str = String::from_utf8_lossy(&bytes);
            return Err(Error::api_with_context(
                format!("TTS API error ({}): {}", status, body_str),
                ErrorContext::new(),
            ));
        }
        let format = options
            .response_format
            .as_deref()
            .map(|s| types::AudioFormat::from_str(s).unwrap_or(types::AudioFormat::Mp3))
            .unwrap_or(types::AudioFormat::Mp3);
        Ok(AudioOutput {
            data: bytes.to_vec(),
            format,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

pub struct TtsClientBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    endpoint_path: Option<String>,
    #[allow(dead_code)]
    timeout_secs: u64,
    manifest: Option<ProtocolManifest>,
}

impl TtsClientBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            endpoint_path: None,
            timeout_secs: 60,
            manifest: None,
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

    pub async fn build(self) -> Result<TtsClient> {
        let model = self
            .model
            .ok_or_else(|| Error::configuration("Model must be specified"))?;
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| Error::configuration("API key required"))?;
        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.openai.com".to_string());
        let endpoint_path = normalize_endpoint_path(
            self.endpoint_path
                .unwrap_or_else(|| "/v1/audio/speech".to_string()),
        );
        let transport =
            build_ancillary_transport(self.manifest.as_ref(), &base_url, &model, &api_key)?;
        Ok(TtsClient {
            transport,
            model,
            endpoint_path,
        })
    }
}

impl Default for TtsClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}
