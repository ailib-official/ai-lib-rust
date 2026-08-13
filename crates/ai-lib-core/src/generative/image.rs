//! Experimental image generation via manifest L-Exec (ALR-GEN-002).

use super::{adapter_name, require_generative_endpoint, KEY_IMAGE_GENERATION};
use crate::credentials::resolve_credential;
use crate::protocol::ProtocolManifest;
use crate::transport::HttpTransport;
use crate::types::{GeneratedImage, ImageGenerationRequest, ImageGenerationResult};
use crate::{Error, ErrorContext, Result};

/// Experimental image generation client (capability: `image_generation`).
pub struct ImageGenerationClient {
    transport: HttpTransport,
    model: String,
    endpoint_path: String,
    adapter: String,
}

impl ImageGenerationClient {
    /// Build from manifest: capability gate + `endpoints.image_generation`.
    pub fn from_manifest(manifest: &ProtocolManifest, model: &str) -> Result<Self> {
        let ep = require_generative_endpoint(manifest, model, KEY_IMAGE_GENERATION)?;
        let cred = resolve_credential(manifest, None);
        let secret = cred.secret().ok_or_else(|| {
            Error::configuration(format!(
                "API key required for image_generation (provider={}; tried {:?})",
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
            adapter: adapter_name(ep).to_string(),
        })
    }

    pub fn endpoint_path(&self) -> &str {
        &self.endpoint_path
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub async fn generate(&self, request: ImageGenerationRequest) -> Result<ImageGenerationResult> {
        if request.model != self.model {
            return Err(Error::configuration(format!(
                "request model `{}` != client model `{}`",
                request.model, self.model
            )));
        }
        let body = match self.adapter.as_str() {
            "dashscope" => dashscope_image_body(&request),
            _ => openai_image_body(&request),
        };
        let resp = self
            .transport
            .execute_stream_response("POST", &self.endpoint_path, &body, None, false)
            .await
            .map_err(|e| {
                Error::network_with_context(
                    format!("image_generation request failed: {e}"),
                    ErrorContext::new().with_source("generative.image"),
                )
            })?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            Error::network_with_context(
                format!("Failed to read image_generation response: {e}"),
                ErrorContext::new(),
            )
        })?;
        if !status.is_success() {
            return Err(Error::api_with_context(
                format!("image_generation API error ({status}): {text}"),
                ErrorContext::new(),
            ));
        }
        let json: serde_json::Value = serde_json::from_str(&text)?;
        match self.adapter.as_str() {
            "dashscope" => parse_dashscope_image(&self.model, &json),
            _ => parse_openai_image(&self.model, &json),
        }
    }
}

fn openai_image_body(req: &ImageGenerationRequest) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": req.model,
        "prompt": req.prompt,
    });
    if let Some(size) = &req.size {
        body["size"] = serde_json::Value::String(size.clone());
    }
    if let Some(n) = req.n {
        body["n"] = serde_json::json!(n);
    }
    if let Some(fmt) = &req.response_format {
        body["response_format"] = serde_json::Value::String(fmt.clone());
    }
    body
}

/// DashScope native multimodal-generation shape (PT-GEN-003).
fn dashscope_image_body(req: &ImageGenerationRequest) -> serde_json::Value {
    serde_json::json!({
        "model": req.model,
        "input": {
            "messages": [{
                "role": "user",
                "content": [{ "text": req.prompt }]
            }]
        }
    })
}

fn parse_openai_image(model: &str, json: &serde_json::Value) -> Result<ImageGenerationResult> {
    let mut images = Vec::new();
    if let Some(arr) = json.get("data").and_then(|v| v.as_array()) {
        for item in arr {
            images.push(GeneratedImage {
                url: item.get("url").and_then(|v| v.as_str()).map(str::to_string),
                b64_json: item
                    .get("b64_json")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            });
        }
    }
    Ok(ImageGenerationResult {
        model: model.to_string(),
        images,
    })
}

fn parse_dashscope_image(model: &str, json: &serde_json::Value) -> Result<ImageGenerationResult> {
    // Best-effort: pull first image URL from common DashScope response shapes.
    let mut images = Vec::new();
    if let Some(url) = json
        .pointer("/output/choices/0/message/content/0/image")
        .and_then(|v| v.as_str())
    {
        images.push(GeneratedImage {
            url: Some(url.to_string()),
            b64_json: None,
        });
    } else if let Some(url) = json
        .pointer("/output/results/0/url")
        .and_then(|v| v.as_str())
    {
        images.push(GeneratedImage {
            url: Some(url.to_string()),
            b64_json: None,
        });
    }
    Ok(ImageGenerationResult {
        model: model.to_string(),
        images,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_and_dashscope_bodies_differ() {
        let req = ImageGenerationRequest::new("m", "a cat");
        let oai = openai_image_body(&req);
        let ds = dashscope_image_body(&req);
        assert_eq!(oai["prompt"], "a cat");
        assert!(oai.get("input").is_none());
        assert!(ds.get("prompt").is_none());
        assert_eq!(ds["input"]["messages"][0]["content"][0]["text"], "a cat");
    }
}
