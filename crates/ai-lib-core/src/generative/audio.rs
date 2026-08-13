//! Experimental STT / TTS via manifest L-Exec (ALR-GEN-002).
//!
//! OpenAI adapter shapes; other adapters return configuration errors until
//! a dialect is declared (Qwen Stage D sample is image-only today).

use super::{adapter_name, require_generative_endpoint, KEY_SPEECH_TO_TEXT, KEY_TEXT_TO_SPEECH};
use crate::credentials::resolve_credential;
use crate::protocol::ProtocolManifest;
use crate::transport::HttpTransport;
use crate::types::{
    SpeechToTextRequest, SpeechToTextResult, TextToSpeechRequest, TextToSpeechResult,
};
use crate::{Error, ErrorContext, Result};
use std::path::Path;

/// Experimental speech-to-text client (capability: `speech_to_text`).
pub struct SpeechToTextClient {
    transport: HttpTransport,
    model: String,
    endpoint_path: String,
    adapter: String,
}

impl SpeechToTextClient {
    pub fn from_manifest(manifest: &ProtocolManifest, model: &str) -> Result<Self> {
        let ep = require_generative_endpoint(manifest, model, KEY_SPEECH_TO_TEXT)?;
        let adapter = adapter_name(ep).to_string();
        if adapter != "openai" {
            return Err(Error::configuration(format!(
                "speech_to_text adapter `{adapter}` not implemented in ALR-GEN-002 (openai only)"
            )));
        }
        let cred = resolve_credential(manifest, None);
        let secret = cred.secret().ok_or_else(|| {
            Error::configuration(format!(
                "API key required for speech_to_text (provider={}; tried {:?})",
                crate::credentials::provider_id(manifest),
                cred.required_envs
                    .iter()
                    .chain(cred.conventional_envs.iter())
                    .cloned()
                    .collect::<Vec<_>>()
            ))
        })?;
        let base_url = manifest.get_base_url().to_string();
        let transport = HttpTransport::new_with_base_url_and_credential(
            manifest,
            model,
            Some(&base_url),
            Some(secret),
        )?;
        Ok(Self {
            transport,
            model: model.to_string(),
            endpoint_path: ep.path.clone(),
            adapter,
        })
    }

    pub fn endpoint_path(&self) -> &str {
        &self.endpoint_path
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub async fn transcribe(&self, request: SpeechToTextRequest) -> Result<SpeechToTextResult> {
        if request.model != self.model {
            return Err(Error::configuration(format!(
                "request model `{}` != client model `{}`",
                request.model, self.model
            )));
        }
        let bytes = std::fs::read(Path::new(&request.audio_source)).map_err(|e| {
            Error::configuration(format!(
                "failed to read audio_source `{}`: {e}",
                request.audio_source
            ))
        })?;
        let file_name = Path::new(&request.audio_source)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("audio.wav")
            .to_string();
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")
            .map_err(|e| Error::configuration(format!("Invalid mime: {e}")))?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone());
        if let Some(lang) = &request.language {
            form = form.text("language", lang.clone());
        }
        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }
        let resp = self
            .transport
            .execute_multipart_response(&self.endpoint_path, form)
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("speech_to_text request failed: {e}"),
                    ErrorContext::new().with_source("generative.stt"),
                )
            })?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            Error::network_with_context(
                format!("Failed to read speech_to_text response: {e}"),
                ErrorContext::new(),
            )
        })?;
        if !status.is_success() {
            return Err(Error::api_with_context(
                format!("speech_to_text API error ({status}): {body}"),
                ErrorContext::new(),
            ));
        }
        let json: serde_json::Value = serde_json::from_str(&body)?;
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(SpeechToTextResult {
            model: self.model.clone(),
            text,
        })
    }
}

/// Experimental text-to-speech client (capability: `text_to_speech`).
pub struct TextToSpeechClient {
    transport: HttpTransport,
    model: String,
    endpoint_path: String,
    adapter: String,
}

impl TextToSpeechClient {
    pub fn from_manifest(manifest: &ProtocolManifest, model: &str) -> Result<Self> {
        let ep = require_generative_endpoint(manifest, model, KEY_TEXT_TO_SPEECH)?;
        let adapter = adapter_name(ep).to_string();
        if adapter != "openai" {
            return Err(Error::configuration(format!(
                "text_to_speech adapter `{adapter}` not implemented in ALR-GEN-002 (openai only)"
            )));
        }
        let cred = resolve_credential(manifest, None);
        let secret = cred.secret().ok_or_else(|| {
            Error::configuration(format!(
                "API key required for text_to_speech (provider={}; tried {:?})",
                crate::credentials::provider_id(manifest),
                cred.required_envs
                    .iter()
                    .chain(cred.conventional_envs.iter())
                    .cloned()
                    .collect::<Vec<_>>()
            ))
        })?;
        let base_url = manifest.get_base_url().to_string();
        let transport = HttpTransport::new_with_base_url_and_credential(
            manifest,
            model,
            Some(&base_url),
            Some(secret),
        )?;
        Ok(Self {
            transport,
            model: model.to_string(),
            endpoint_path: ep.path.clone(),
            adapter,
        })
    }

    pub fn endpoint_path(&self) -> &str {
        &self.endpoint_path
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub async fn synthesize(&self, request: TextToSpeechRequest) -> Result<TextToSpeechResult> {
        if request.model != self.model {
            return Err(Error::configuration(format!(
                "request model `{}` != client model `{}`",
                request.model, self.model
            )));
        }
        let mut body = serde_json::json!({
            "model": request.model,
            "input": request.input,
        });
        if let Some(voice) = &request.voice {
            body["voice"] = serde_json::Value::String(voice.clone());
        }
        if let Some(fmt) = &request.response_format {
            body["response_format"] = serde_json::Value::String(fmt.clone());
        }
        let resp = self
            .transport
            .execute_stream_response("POST", &self.endpoint_path, &body, None, false)
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("text_to_speech request failed: {e}"),
                    ErrorContext::new().with_source("generative.tts"),
                )
            })?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let bytes = resp.bytes().await.map_err(|e| {
            Error::network_with_context(
                format!("Failed to read text_to_speech response: {e}"),
                ErrorContext::new(),
            )
        })?;
        if !status.is_success() {
            let body = String::from_utf8_lossy(&bytes);
            return Err(Error::api_with_context(
                format!("text_to_speech API error ({status}): {body}"),
                ErrorContext::new(),
            ));
        }
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        Ok(TextToSpeechResult {
            model: self.model.clone(),
            audio_base64: Some(STANDARD.encode(bytes)),
            content_type,
        })
    }
}
