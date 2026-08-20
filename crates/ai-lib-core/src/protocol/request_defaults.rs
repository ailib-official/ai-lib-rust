//! Experimental `request_defaults` apply (ALR-REQ-DEFAULTS-RUNTIME-001 / GOV-007).
//!
//! # Experimental request_defaults apply
//!
//! Single production entry for overlaying `metadata.models.<id>.request_defaults`
//! onto an OpenAI-compatible request body (max_tokens cap + reasoning_effort).
//! Manifest-first: omit / missing defaults → no-op. No per-model hardcoded tables.

use serde_json::{Map, Number, Value};
use std::collections::HashMap;

/// Options for [`apply_request_defaults`].
#[derive(Debug, Clone, Copy)]
pub struct RequestDefaultsOptions<'a> {
    /// Host think toggle. When true, prefer `*_thinking` caps / effort.
    pub thinking_enabled: bool,
    /// Wire model id (unused for caps; reserved for future effort rules).
    pub model_id: Option<&'a str>,
    /// Manifest / catalog `request_defaults` object. `None` → no-op.
    pub request_defaults: Option<&'a Value>,
}

impl<'a> RequestDefaultsOptions<'a> {
    /// Build options from an optional defaults object.
    pub fn new(thinking_enabled: bool, request_defaults: Option<&'a Value>) -> Self {
        Self {
            thinking_enabled,
            model_id: None,
            request_defaults,
        }
    }

    /// Attach model id (forward-compatible).
    pub fn with_model_id(mut self, model_id: &'a str) -> Self {
        self.model_id = Some(model_id);
        self
    }
}

/// Look up `metadata.models.<model_id>.request_defaults` from a flattened manifest `extra` map.
pub fn request_defaults_from_extra<'a>(
    extra: &'a HashMap<String, Value>,
    model_id: &str,
) -> Option<&'a Value> {
    let entry = extra.get("metadata")?.get("models")?.get(model_id)?;
    entry.get("request_defaults")
}

/// Positive integer from a JSON number field, if present and > 0.
fn positive_u64(v: Option<&Value>) -> Option<u64> {
    let n = v?.as_u64().or_else(|| v?.as_f64().map(|f| f as u64))?;
    if n > 0 {
        Some(n)
    } else {
        None
    }
}

/// Select max_tokens cap from defaults for the think toggle.
pub fn max_tokens_cap_from_defaults(defaults: &Value, thinking_enabled: bool) -> Option<u64> {
    if thinking_enabled {
        if let Some(c) = positive_u64(defaults.get("max_tokens_cap_thinking")) {
            return Some(c);
        }
    }
    positive_u64(defaults.get("max_tokens_cap"))
}

/// Select safe input token budget from defaults (for hosts / assemble capacity).
pub fn safe_input_tokens_from_defaults(defaults: &Value, thinking_enabled: bool) -> Option<u64> {
    if thinking_enabled {
        if let Some(c) = positive_u64(defaults.get("safe_input_tokens_thinking")) {
            return Some(c);
        }
    }
    positive_u64(defaults.get("safe_input_tokens"))
}

/// Select `reasoning_effort` wire string from defaults for the think toggle.
pub fn reasoning_effort_from_defaults(defaults: &Value, thinking_enabled: bool) -> Option<String> {
    let re = defaults.get("reasoning_effort")?;
    if !re.is_object() {
        return None;
    }
    if thinking_enabled {
        if let Some(s) = re.get("when_thinking_on").and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
    } else if let Some(s) = re.get("when_thinking_off").and_then(|v| v.as_str()) {
        return Some(s.to_string());
    }
    re.get("default")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn body_as_object(body: &mut Value) -> Option<&mut Map<String, Value>> {
    match body {
        Value::Object(m) => Some(m),
        _ => None,
    }
}

fn read_positive_token_field(obj: &Map<String, Value>, key: &str) -> Option<u64> {
    positive_u64(obj.get(key))
}

/// Apply Experimental `request_defaults` onto an OpenAI-compat request body.
///
/// - Caps `max_tokens` (and clears `max_completion_tokens` to avoid dual-field TPM bugs).
/// - Fills `reasoning_effort` only when absent on the body and present in defaults.
/// - No-op when `request_defaults` is `None` or empty of usable keys.
pub fn apply_request_defaults(body: &mut Value, opts: &RequestDefaultsOptions<'_>) {
    let Some(defaults) = opts.request_defaults else {
        return;
    };
    if !defaults.is_object() {
        return;
    }

    let Some(obj) = body_as_object(body) else {
        return;
    };

    if let Some(cap) = max_tokens_cap_from_defaults(defaults, opts.thinking_enabled) {
        let prev = read_positive_token_field(obj, "max_tokens")
            .or_else(|| read_positive_token_field(obj, "max_completion_tokens"));
        let budget = match prev {
            Some(p) => p.min(cap),
            None => cap,
        };
        obj.insert(
            "max_tokens".to_string(),
            Value::Number(Number::from(budget)),
        );
        obj.remove("max_completion_tokens");
    }

    let has_effort = obj
        .get("reasoning_effort")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    if !has_effort {
        if let Some(effort) = reasoning_effort_from_defaults(defaults, opts.thinking_enabled) {
            obj.insert("reasoning_effort".to_string(), Value::String(effort));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qwen_defaults() -> Value {
        json!({
            "status": "experimental",
            "max_tokens_cap": 2048,
            "max_tokens_cap_thinking": 4096,
            "safe_input_tokens": 3500,
            "safe_input_tokens_thinking": 2800,
            "reasoning_effort": {
                "when_thinking_off": "none",
                "when_thinking_on": "default"
            }
        })
    }

    #[test]
    fn omit_defaults_is_noop() {
        let mut body = json!({ "model": "x", "max_tokens": 9999 });
        apply_request_defaults(&mut body, &RequestDefaultsOptions::new(true, None));
        assert_eq!(body["max_tokens"], 9999);
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn think_on_uses_thinking_cap_and_effort() {
        let rd = qwen_defaults();
        let mut body = json!({ "model": "qwen/qwen3.6-27b", "max_tokens": 8000 });
        apply_request_defaults(
            &mut body,
            &RequestDefaultsOptions::new(true, Some(&rd)).with_model_id("qwen/qwen3.6-27b"),
        );
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["reasoning_effort"], "default");
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn think_off_uses_baseline_cap_and_effort() {
        let rd = qwen_defaults();
        let mut body = json!({ "model": "qwen/qwen3.6-27b", "max_tokens": 8000 });
        apply_request_defaults(&mut body, &RequestDefaultsOptions::new(false, Some(&rd)));
        assert_eq!(body["max_tokens"], 2048);
        assert_eq!(body["reasoning_effort"], "none");
    }

    #[test]
    fn missing_prev_max_tokens_sets_cap() {
        let rd = qwen_defaults();
        let mut body = json!({ "model": "m" });
        apply_request_defaults(&mut body, &RequestDefaultsOptions::new(true, Some(&rd)));
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn does_not_overwrite_existing_reasoning_effort() {
        let rd = qwen_defaults();
        let mut body = json!({
            "model": "m",
            "max_tokens": 1000,
            "reasoning_effort": "low"
        });
        apply_request_defaults(&mut body, &RequestDefaultsOptions::new(true, Some(&rd)));
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(body["max_tokens"], 1000);
    }

    #[test]
    fn strips_max_completion_tokens() {
        let rd = qwen_defaults();
        let mut body = json!({
            "model": "m",
            "max_completion_tokens": 5000
        });
        apply_request_defaults(&mut body, &RequestDefaultsOptions::new(false, Some(&rd)));
        assert_eq!(body["max_tokens"], 2048);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn safe_input_tokens_selector() {
        let rd = qwen_defaults();
        assert_eq!(safe_input_tokens_from_defaults(&rd, true), Some(2800));
        assert_eq!(safe_input_tokens_from_defaults(&rd, false), Some(3500));
    }

    #[test]
    fn lookup_from_extra() {
        let mut extra = HashMap::new();
        extra.insert(
            "metadata".into(),
            json!({
                "models": {
                    "qwen/qwen3.6-27b": {
                        "request_defaults": { "max_tokens_cap": 2048 }
                    }
                }
            }),
        );
        let rd = request_defaults_from_extra(&extra, "qwen/qwen3.6-27b").expect("rd");
        assert_eq!(rd["max_tokens_cap"], 2048);
        assert!(request_defaults_from_extra(&extra, "missing").is_none());
    }

    #[test]
    fn compile_request_hooks_apply_from_override() {
        use crate::protocol::{ProtocolManifest, UnifiedRequest};

        let manifest: ProtocolManifest = serde_json::from_value(json!({
            "id": "test",
            "protocol_version": "2.0.0",
            "endpoint": { "base_url": "https://example.com" },
            "capabilities": { "streaming": true, "tools": false, "vision": false },
            "status": "stable",
            "category": "ai_provider",
            "official_url": "https://example.com",
            "support_contact": "dev@example.com",
            "parameter_mappings": {
                "model": "model",
                "messages": "messages",
                "max_tokens": "max_tokens",
                "stream": "stream"
            }
        }))
        .expect("manifest");

        let req = UnifiedRequest {
            operation: "chat".into(),
            model: "qwen/qwen3.6-27b".into(),
            max_tokens: Some(8000),
            thinking_enabled: Some(true),
            request_defaults: Some(qwen_defaults()),
            ..Default::default()
        };
        let body = manifest.compile_request(&req).expect("compile");
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["reasoning_effort"], "default");

        let req_off = UnifiedRequest {
            thinking_enabled: Some(false),
            request_defaults: Some(qwen_defaults()),
            max_tokens: Some(8000),
            model: "qwen/qwen3.6-27b".into(),
            operation: "chat".into(),
            ..Default::default()
        };
        let body_off = manifest.compile_request(&req_off).expect("compile");
        assert_eq!(body_off["max_tokens"], 2048);
        assert_eq!(body_off["reasoning_effort"], "none");
    }
}
