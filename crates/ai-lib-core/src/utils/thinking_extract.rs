//! OpenAI-compatible thinking / reasoning field extraction (ALR-RSN-001).
//! 中文：从 delta/message 结构化字段提取思考文本；单一别名表，供 Driver 与 event_map 共用（GOV-007）。

use serde_json::Value;

/// Wire keys observed across OpenAI-compatible reasoners / proxies.
/// Order is preference when multiple keys appear on the same object.
pub const OPENAI_COMPAT_THINKING_KEYS: &[&str] = &[
    "reasoning_content",
    "reasoning",
    "thinking",
    "thought",
    "reasoning_text",
    "analysis",
];

/// First non-empty string among `keys` on a JSON object.
pub fn first_nonempty_string_field(obj: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(s) = obj.get(*key).and_then(|v| v.as_str()) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

/// Thinking text from `choices[0].delta.*` (streaming).
pub fn thinking_from_openai_compat_delta(frame: &Value) -> Option<String> {
    let delta = frame.pointer("/choices/0/delta")?;
    first_nonempty_string_field(delta, OPENAI_COMPAT_THINKING_KEYS)
}

/// Thinking text from `choices[0].message.*` (non-streaming).
pub fn thinking_from_openai_compat_message(frame: &Value) -> Option<String> {
    let msg = frame.pointer("/choices/0/message")?;
    first_nonempty_string_field(msg, OPENAI_COMPAT_THINKING_KEYS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn delta_prefers_reasoning_content() {
        let frame = json!({
            "choices": [{"delta": {
                "reasoning_content": "a",
                "thinking": "b",
                "content": "c"
            }}]
        });
        assert_eq!(
            thinking_from_openai_compat_delta(&frame).as_deref(),
            Some("a")
        );
    }

    #[test]
    fn delta_alias_thinking() {
        let frame = json!({"choices":[{"delta":{"thinking":"plan"}}]});
        assert_eq!(
            thinking_from_openai_compat_delta(&frame).as_deref(),
            Some("plan")
        );
    }

    #[test]
    fn message_reasoning_not_content() {
        let frame = json!({
            "choices":[{"message":{"content":"","reasoning_content":"only think"}}]
        });
        assert_eq!(
            thinking_from_openai_compat_message(&frame).as_deref(),
            Some("only think")
        );
    }
}
