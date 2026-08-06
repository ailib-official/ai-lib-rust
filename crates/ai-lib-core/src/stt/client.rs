//! STT (Speech-to-Text) client.
//!
//! HTTP uses the same [`HttpTransport`] stack as chat ([GOV-007]).

use super::types::{SttOptions, Transcription};
use crate::transport::HttpTransport;
use crate::{Error, ErrorContext, Result};

/// Client for speech-to-text transcription.
pub struct SttClient {
    transport: HttpTransport,
    model: String,
    endpoint_path: String,
}

impl SttClient {
    pub fn builder() -> SttClientBuilder {
        SttClientBuilder::new()
    }

    pub async fn transcribe(&self, audio: &[u8], options: &SttOptions) -> Result<Transcription> {
        let part = reqwest::multipart::Part::bytes(audio.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| Error::configuration(format!("Invalid mime: {}", e)))?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone());
        if let Some(lang) = &options.language {
            form = form.text("language", lang.clone());
        }
        if let Some(prompt) = &options.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(temp) = options.temperature {
            form = form.text("temperature", temp.to_string());
        }
        if let Some(rf) = &options.response_format {
            form = form.text("response_format", rf.clone());
        }
        let resp = self
            .transport
            .execute_multipart_response(&self.endpoint_path, form)
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("STT request failed: {e}"),
                    ErrorContext::new().with_source("stt"),
                )
            })?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            Error::network_with_context(
                format!("Failed to read STT response: {e}"),
                ErrorContext::new(),
            )
        })?;
        if !status.is_success() {
            return Err(Error::api_with_context(
                format!("STT API error ({}): {}", status, body),
                ErrorContext::new(),
            ));
        }
        let json: serde_json::Value = serde_json::from_str(&body)?;
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(Transcription {
            text,
            language: json
                .get("language")
                .and_then(|v| v.as_str())
                .map(String::from),
            confidence: None,
            segments: None,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

fn ensure_abs_path(path: String) -> String {
    if path.starts_with('/') {
        path
    } else {
        format!("/{path}")
    }
}

pub struct SttClientBuilder {
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    endpoint_path: Option<String>,
    #[allow(dead_code)]
    timeout_secs: u64,
}

impl SttClientBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            endpoint_path: None,
            timeout_secs: 60,
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

    pub async fn build(self) -> Result<SttClient> {
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
        let endpoint_path = ensure_abs_path(
            self.endpoint_path
                .unwrap_or_else(|| "/v1/audio/transcriptions".to_string()),
        );
        let transport = HttpTransport::with_explicit_bearer(&base_url, &model, &api_key)?;
        Ok(SttClient {
            transport,
            model,
            endpoint_path,
        })
    }
}

impl Default for SttClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}
