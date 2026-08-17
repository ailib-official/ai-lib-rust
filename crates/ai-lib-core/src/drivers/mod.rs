//! Provider 驱动抽象层 — 通过 trait 实现多厂商 API 适配的动态分发
//!
//! Provider driver abstraction layer implementing the ProviderContract specification.
//! Uses `Box<dyn ProviderDriver>` for runtime polymorphism, enabling the same client
//! code to work with OpenAI, Anthropic, Gemini, and any OpenAI-compatible provider.

pub mod anthropic;
pub mod gemini;

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::error::Error;
use crate::protocol::v2::capabilities::Capability;
use crate::protocol::v2::manifest::ApiStyle;
use crate::protocol::ProtocolError;
use crate::types::events::StreamingEvent;
use crate::types::execution_result::ExecutionUsage;
use crate::types::message::{ContentBlock, Message, MessageContent};

pub use anthropic::AnthropicDriver;
pub use gemini::GeminiDriver;

/// Unified HTTP request representation for provider communication.
#[derive(Debug, Clone)]
pub struct DriverRequest {
    /// Target URL (base_url + chat_path).
    pub url: String,
    /// HTTP method (POST for chat, GET for models).
    pub method: String,
    /// Request headers.
    pub headers: HashMap<String, String>,
    /// Serialized JSON request body.
    pub body: Value,
    /// Whether streaming is requested.
    pub stream: bool,
}

/// Unified chat response from provider.
#[derive(Debug, Clone)]
pub struct DriverResponse {
    /// Extracted text content.
    pub content: Option<String>,
    /// Extended thinking / reasoning text when present (not user-facing answer).
    pub thinking: Option<String>,
    /// Finish reason normalized to AI-Protocol standard.
    pub finish_reason: Option<String>,
    /// Token usage statistics.
    pub usage: Option<UsageInfo>,
    /// Tool calls if any.
    pub tool_calls: Vec<Value>,
    /// Raw provider response for debugging.
    pub raw: Value,
}

/// Token usage information.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// OpenAI-style `completion_tokens_details.reasoning_tokens`.
    pub reasoning_tokens: Option<u64>,
    /// Anthropic cache read / creation input tokens (when present).
    pub cache_read_tokens: Option<u64>,
    pub cache_creation_tokens: Option<u64>,
}

impl From<UsageInfo> for ExecutionUsage {
    fn from(u: UsageInfo) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u.reasoning_tokens,
            cache_read_tokens: u.cache_read_tokens,
            cache_creation_tokens: u.cache_creation_tokens,
        }
    }
}

/// Core trait for provider-specific API adaptation.
///
/// Each provider API style (OpenAI, Anthropic, Gemini) has a concrete implementation.
/// The trait is object-safe and supports dynamic dispatch via `Box<dyn ProviderDriver>`.
///
/// # Design Notes
///
/// Inspired by `sqlx::Database` — the trait defines the contract, concrete types
/// implement the transformations. The runtime selects the correct driver based on
/// the manifest's `api_style` or `provider_contract`.
#[async_trait]
pub trait ProviderDriver: Send + Sync + std::fmt::Debug {
    /// Unique provider identifier (matches manifest `id`).
    fn provider_id(&self) -> &str;

    /// API style this driver implements.
    fn api_style(&self) -> ApiStyle;

    /// Build a provider-specific HTTP request from unified parameters.
    fn build_request(
        &self,
        messages: &[Message],
        model: &str,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
        stream: bool,
        extra: Option<&Value>,
    ) -> Result<DriverRequest, Error>;

    /// Parse a non-streaming response into unified format.
    fn parse_response(&self, body: &Value) -> Result<DriverResponse, Error>;

    /// Parse a single streaming SSE/NDJSON frame into zero or more unified events.
    ///
    /// A frame may carry both thinking and content (OpenAI-compat reasoners). Callers
    /// MUST drain the full `Vec` — do not assume at most one event (ALR-RSN-001).
    fn parse_stream_event(&self, data: &str) -> Result<Vec<StreamingEvent>, Error>;

    /// Get the list of capabilities this driver supports.
    fn supported_capabilities(&self) -> &[Capability];

    /// Check if the done signal has been received in streaming.
    fn is_stream_done(&self, data: &str) -> bool;
}

/// OpenAI-compatible driver — works for OpenAI, DeepSeek, Moonshot, Zhipu, etc.
#[derive(Debug)]
pub struct OpenAiDriver {
    provider_id: String,
    capabilities: Vec<Capability>,
}

impl OpenAiDriver {
    pub fn new(provider_id: impl Into<String>, capabilities: Vec<Capability>) -> Self {
        Self {
            provider_id: provider_id.into(),
            capabilities,
        }
    }
}

/// Merge OpenAI- and Anthropic-flavored token shapes inside a `usage` object (OpenAI
/// `choices[0].*` envelope). Aligns with browser/TS/Go "unified usage" (ARCH-003).
fn parse_openai_usage_value(u: &Value) -> UsageInfo {
    let flat = |key: &str| -> u64 { u.get(key).and_then(|v| v.as_u64()).unwrap_or(0) };
    let nested = |outer: &str, inner: &str| -> u64 {
        u.get(outer)
            .and_then(|d| d.get(inner))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    let first_nonzero = |vals: &[u64]| -> u64 { *vals.iter().find(|&&v| v != 0).unwrap_or(&0) };

    let prompt_tokens = first_nonzero(&[flat("prompt_tokens"), flat("input_tokens")]);
    let completion_tokens = first_nonzero(&[flat("completion_tokens"), flat("output_tokens")]);
    let mut total_tokens = flat("total_tokens");
    if total_tokens == 0 && (prompt_tokens > 0 || completion_tokens > 0) {
        total_tokens = prompt_tokens + completion_tokens;
    }

    let reason_sum = first_nonzero(&[
        flat("reasoning_tokens"),
        nested("completion_tokens_details", "reasoning_tokens"),
    ]);
    let cache_read = first_nonzero(&[
        flat("cache_read_tokens"),
        nested("prompt_tokens_details", "cached_tokens"),
        flat("cache_read_input_tokens"),
    ]);
    let cache_create = first_nonzero(&[
        flat("cache_creation_tokens"),
        flat("cache_creation_input_tokens"),
        flat("cache_write_tokens"),
    ]);

    UsageInfo {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        reasoning_tokens: (reason_sum > 0).then_some(reason_sum),
        cache_read_tokens: (cache_read > 0).then_some(cache_read),
        cache_creation_tokens: (cache_create > 0).then_some(cache_create),
    }
}

#[async_trait]
impl ProviderDriver for OpenAiDriver {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn api_style(&self) -> ApiStyle {
        ApiStyle::OpenAiCompatible
    }

    fn build_request(
        &self,
        messages: &[Message],
        model: &str,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
        stream: bool,
        extra: Option<&Value>,
    ) -> Result<DriverRequest, Error> {
        for m in messages {
            if let MessageContent::Blocks(blocks) = &m.content {
                for block in blocks {
                    if matches!(block, ContentBlock::Document { .. }) {
                        return Err(Error::Protocol(ProtocolError::ValidationError(
                            "OpenAI-compatible driver does not encode document blocks; use Anthropic or Gemini".into(),
                        )));
                    }
                }
            }
        }

        let oai_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                let role = serde_json::to_value(&m.role).unwrap_or(Value::String("user".into()));
                let content = match &m.content {
                    MessageContent::Text(s) => Value::String(s.clone()),
                    MessageContent::Blocks(_) => {
                        serde_json::to_value(&m.content).unwrap_or(Value::Null)
                    }
                };
                let mut obj = serde_json::json!({ "role": role, "content": content });
                // OpenAI API requires tool_call_id for role "tool"
                if matches!(m.role, crate::types::message::MessageRole::Tool) {
                    if let Some(ref id) = m.tool_call_id {
                        obj["tool_call_id"] = Value::String(id.clone());
                    }
                }
                obj
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": oai_messages,
            "stream": stream,
        });

        if let Some(t) = temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(mt) = max_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }
        if let Some(Value::Object(map)) = extra {
            for (k, v) in map {
                body[k] = v.clone();
            }
        }

        Ok(DriverRequest {
            url: String::new(), // URL is set by the client layer from manifest
            method: "POST".into(),
            headers: HashMap::new(),
            body,
            stream,
        })
    }

    fn parse_response(&self, body: &Value) -> Result<DriverResponse, Error> {
        let content = body
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .map(String::from);
        let thinking = crate::utils::thinking_from_openai_compat_message(body);
        let finish_reason = body
            .pointer("/choices/0/finish_reason")
            .and_then(|v| v.as_str())
            .map(String::from);
        let usage = body.get("usage").map(parse_openai_usage_value);
        let tool_calls = body
            .pointer("/choices/0/message/tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        Ok(DriverResponse {
            content,
            thinking,
            finish_reason,
            usage,
            tool_calls,
            raw: body.clone(),
        })
    }

    fn parse_stream_event(&self, data: &str) -> Result<Vec<StreamingEvent>, Error> {
        if data.trim().is_empty() || self.is_stream_done(data) {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_str(data).map_err(|e| {
            Error::Protocol(ProtocolError::ValidationError(format!(
                "Failed to parse SSE data: {}",
                e
            )))
        })?;

        let mut out = Vec::new();

        // Thinking first when co-present with content (order only — never drop content).
        if let Some(thinking) = crate::utils::thinking_from_openai_compat_delta(&v) {
            out.push(StreamingEvent::ThinkingDelta {
                thinking,
                tool_consideration: None,
            });
        }

        if let Some(content) = v
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
        {
            if !content.is_empty() {
                out.push(StreamingEvent::PartialContentDelta {
                    content: content.to_string(),
                    sequence_id: None,
                });
            }
        }

        if out.is_empty() {
            if let Some(reason) = v
                .pointer("/choices/0/finish_reason")
                .and_then(|r| r.as_str())
            {
                out.push(StreamingEvent::StreamEnd {
                    finish_reason: Some(reason.to_string()),
                });
            }
        }

        Ok(out)
    }

    fn supported_capabilities(&self) -> &[Capability] {
        &self.capabilities
    }

    fn is_stream_done(&self, data: &str) -> bool {
        data.trim() == "[DONE]"
    }
}

/// Factory function to create the appropriate driver from an API style.
///
/// The `Custom` style falls back to OpenAI-compatible, which covers most
/// providers that follow the OpenAI chat completions format (DeepSeek,
/// Moonshot, Zhipu, etc.).
pub fn create_driver(
    api_style: ApiStyle,
    provider_id: &str,
    capabilities: Vec<Capability>,
) -> Box<dyn ProviderDriver> {
    match api_style {
        ApiStyle::OpenAiCompatible | ApiStyle::Custom => {
            Box::new(OpenAiDriver::new(provider_id, capabilities))
        }
        ApiStyle::AnthropicMessages => Box::new(AnthropicDriver::new(provider_id, capabilities)),
        ApiStyle::GeminiGenerate => Box::new(GeminiDriver::new(provider_id, capabilities)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_driver_build_request() {
        let driver = OpenAiDriver::new("openai", vec![Capability::Text, Capability::Streaming]);
        let messages = vec![Message::user("Hello")];
        let req = driver
            .build_request(&messages, "gpt-4", Some(0.7), Some(1024), true, None)
            .unwrap();
        assert!(req.stream);
        assert_eq!(req.body["model"], "gpt-4");
        assert_eq!(req.body["temperature"], 0.7);
    }

    #[test]
    fn test_openai_driver_parse_response() {
        let driver = OpenAiDriver::new("openai", vec![]);
        let body = serde_json::json!({
            "choices": [{"message": {"content": "Hi there!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp = driver.parse_response(&body).unwrap();
        assert_eq!(resp.content.as_deref(), Some("Hi there!"));
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn test_openai_driver_parse_response_reasoning_tokens() {
        let driver = OpenAiDriver::new("openai", vec![]);
        let body = serde_json::json!({
            "choices": [{"message": {"content": "Hello, world!"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3}
            }
        });
        let resp = driver.parse_response(&body).unwrap();
        let u = resp.usage.expect("usage");
        assert_eq!(u.reasoning_tokens, Some(3));
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
    }

    #[test]
    fn test_openai_driver_parse_stream_reasoning_delta() {
        let driver = OpenAiDriver::new("openai", vec![]);
        let data = r#"{"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]}"#;
        let events = driver.parse_stream_event(data).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamingEvent::ThinkingDelta { thinking, .. } => {
                assert_eq!(thinking, "Let me think...");
            }
            other => panic!("Expected ThinkingDelta, got {:?}", other),
        }
    }

    #[test]
    fn test_openai_driver_parse_stream_same_frame_keeps_both() {
        let driver = OpenAiDriver::new("openai", vec![]);
        // Same frame: thinking first, content must not be dropped (Spider / ALR-RSN-001).
        let data = r#"{"choices":[{"delta":{"thinking":"plan","content":"answer"}}]}"#;
        let events = driver.parse_stream_event(data).unwrap();
        assert_eq!(events.len(), 2);
        match &events[0] {
            StreamingEvent::ThinkingDelta { thinking, .. } => assert_eq!(thinking, "plan"),
            other => panic!("expected ThinkingDelta first, got {:?}", other),
        }
        match &events[1] {
            StreamingEvent::PartialContentDelta { content, .. } => assert_eq!(content, "answer"),
            other => panic!("expected PartialContentDelta second, got {:?}", other),
        }
    }

    #[test]
    fn test_openai_driver_parse_response_keeps_thinking_separate() {
        let driver = OpenAiDriver::new("openai", vec![]);
        let body = serde_json::json!({
            "choices": [{
                "message": {"content": "", "reasoning_content": "only think"},
                "finish_reason": "stop"
            }]
        });
        let resp = driver.parse_response(&body).unwrap();
        assert!(
            resp.content.as_deref().unwrap_or("").is_empty() || resp.content.as_deref() == Some("")
        );
        assert_eq!(resp.thinking.as_deref(), Some("only think"));
    }

    /// Unified `usage` object may carry Anthropic-style keys inside an OpenAI `choices[0].*` envelope.
    #[test]
    fn test_openai_driver_parse_unified_usage_anthropic_keys() {
        let driver = OpenAiDriver::new("openai", vec![]);
        let body = serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 3,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 2
            }
        });
        let resp = driver.parse_response(&body).unwrap();
        let u = resp.usage.expect("usage");
        assert_eq!(u.prompt_tokens, 12);
        assert_eq!(u.completion_tokens, 3);
        assert_eq!(u.total_tokens, 15);
        assert_eq!(u.cache_creation_tokens, Some(5));
        assert_eq!(u.cache_read_tokens, Some(2));
    }

    #[test]
    fn test_openai_driver_parse_stream() {
        let driver = OpenAiDriver::new("openai", vec![]);
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"index":0}]}"#;
        let events = driver.parse_stream_event(data).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamingEvent::PartialContentDelta { content, .. } => {
                assert_eq!(content, "Hello");
            }
            _ => panic!("Expected PartialContentDelta"),
        }
    }

    #[test]
    fn test_openai_driver_rejects_document_blocks() {
        use crate::types::message::{ContentBlock, MessageContent, MessageRole};

        let driver = OpenAiDriver::new("openai", vec![Capability::Text]);
        let messages = vec![Message::with_content(
            MessageRole::User,
            MessageContent::blocks(vec![ContentBlock::document_base64(
                "abc".into(),
                Some("application/pdf".into()),
                None,
            )]),
        )];
        assert!(driver
            .build_request(&messages, "gpt-4o", None, None, false, None)
            .is_err());
    }
}
