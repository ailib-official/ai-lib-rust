//! Text-based tool call parsing for LLMs without reliable native function calling.
//!
//! 文本工具调用解析：适用于不支持或不稳定 native function calling 的 provider。

use super::tool::{ToolCall, ToolDefinition, ToolResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Prompt strategy level (L1 standard / L2 counterexamples / L3 few-shot).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PromptLevel {
    #[default]
    L1,
    L2,
    L3,
}

impl PromptLevel {
    fn parse(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "L2" => Self::L2,
            "L3" => Self::L3,
            _ => Self::L1,
        }
    }
}

/// Configuration for text tool call parsing and prompt generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToolConfig {
    /// Enable lenient parsing (L2-L4 dialect/alias handling).
    #[serde(default)]
    pub lenient_parsing: bool,
    /// Max nesting depth for tool_call blocks.
    #[serde(default = "default_max_depth")]
    pub max_call_depth: u8,
    /// Include counterexample warnings in prompts (L2+).
    #[serde(default = "default_true")]
    pub include_counterexamples: bool,
    /// Prompt strategy level.
    #[serde(default)]
    pub prompt_level: PromptLevel,
    /// Prompt locale: "en" or "zh".
    #[serde(default = "default_locale")]
    pub locale: String,
    /// Preferred JSON key for arguments when normalizing (from manifest args_key).
    #[serde(default)]
    pub args_key: Option<String>,
    /// Manifest-declared L4 dialect tags (`known_dialects`).
    #[serde(default)]
    pub dialects: Vec<KnownDialect>,
}

/// Manifest `known_dialects` entry: XML tag → tool name.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnownDialect {
    pub tag: String,
    #[serde(default)]
    pub map_to: String,
}

fn default_max_depth() -> u8 {
    1
}
fn default_true() -> bool {
    true
}
fn default_locale() -> String {
    "en".to_string()
}

impl Default for TextToolConfig {
    fn default() -> Self {
        Self {
            lenient_parsing: false,
            max_call_depth: 1,
            include_counterexamples: true,
            prompt_level: PromptLevel::L1,
            locale: "en".to_string(),
            args_key: None,
            dialects: Vec::new(),
        }
    }
}

/// Native tool-calling strategy derived from manifest `tool_calling`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeStrategy {
    /// Native API tools only; text fallback rarely needed.
    Full,
    /// Native API tools plus lenient text fallback (e.g. DeepSeek partial).
    Hybrid,
    /// Text-only protocol; do not send native tool specs.
    TextOnly,
}

/// Runtime policy: dispatcher selection + manifest-backed parser.
#[derive(Debug, Clone)]
pub struct ToolCallingPolicy {
    pub native_strategy: NativeStrategy,
    pub parser: StandardTextToolParser,
}

impl ToolCallingPolicy {
    /// Build policy from manifest `tool_calling` JSON (None → conservative text-only defaults).
    pub fn from_tool_calling(tool_calling: Option<&serde_json::Value>) -> Self {
        let parser = tool_calling
            .map(StandardTextToolParser::from_manifest_tool_calling)
            .unwrap_or_else(default_lenient_parser);
        let native_strategy = tool_calling
            .map(infer_native_strategy)
            .unwrap_or(NativeStrategy::TextOnly);
        Self {
            native_strategy,
            parser,
        }
    }

    /// Whether the application should send native tool specs to the provider API.
    pub fn send_native_tool_specs(&self) -> bool {
        matches!(
            self.native_strategy,
            NativeStrategy::Full | NativeStrategy::Hybrid
        )
    }

    /// Whether auto mode should prefer `NativeToolDispatcher` (hybrid text fallback).
    pub fn prefer_native_dispatcher(&self) -> bool {
        self.send_native_tool_specs()
    }
}

fn default_lenient_parser() -> StandardTextToolParser {
    StandardTextToolParser::new(TextToolConfig {
        lenient_parsing: true,
        prompt_level: PromptLevel::L2,
        include_counterexamples: true,
        ..Default::default()
    })
}

fn infer_native_strategy(tool_calling: &serde_json::Value) -> NativeStrategy {
    let native_supported = tool_calling
        .get("native")
        .and_then(|n| n.get("supported"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if !native_supported {
        return NativeStrategy::TextOnly;
    }

    let reliability = tool_calling
        .get("native")
        .and_then(|n| n.get("reliability"))
        .and_then(|v| v.as_str())
        .unwrap_or("unreliable");

    let has_text_fallback = tool_calling
        .get("text_fallback")
        .map(|v| !v.is_null())
        .unwrap_or(false);

    match reliability {
        "full" => NativeStrategy::Full,
        "partial" if has_text_fallback => NativeStrategy::Hybrid,
        "partial" => NativeStrategy::Full,
        _ if has_text_fallback => NativeStrategy::TextOnly,
        _ => NativeStrategy::Full,
    }
}

/// Cross-LLM text tool call parser trait.
pub trait TextToolParser: Send + Sync {
    /// Split LLM response into plain text and structured tool calls.
    fn parse(&self, response_text: &str) -> (String, Vec<ToolCall>);

    /// Generate system prompt instructions for tool use protocol.
    fn prompt_instructions(&self, tools: &[ToolDefinition]) -> String;

    /// Format tool execution results for the next LLM turn.
    fn format_results(&self, results: &[ToolResult]) -> String;
}

/// Default implementation using the AI-Protocol standard `<tool_call>` format.
#[derive(Debug, Clone)]
pub struct StandardTextToolParser {
    config: TextToolConfig,
}

impl StandardTextToolParser {
    pub fn new(config: TextToolConfig) -> Self {
        Self { config }
    }

    /// Build parser config from a provider manifest `tool_calling.text_fallback` block.
    pub fn from_manifest_tool_calling(tool_calling: &serde_json::Value) -> Self {
        let mut config = TextToolConfig {
            lenient_parsing: true,
            prompt_level: PromptLevel::L2,
            ..Default::default()
        };

        if let Some(fallback) = tool_calling.get("text_fallback") {
            if fallback.is_null() {
                // explicit null — no text fallback config
            } else {
                if let Some(level) = fallback.get("prompt_level").and_then(|v| v.as_str()) {
                    config.prompt_level = PromptLevel::parse(level);
                }
                if let Some(key) = fallback.get("args_key").and_then(|v| v.as_str()) {
                    config.args_key = Some(key.to_string());
                }
                if let Some(list) = fallback.get("known_dialects").and_then(|v| v.as_array()) {
                    for entry in list {
                        let tag = entry.get("tag").and_then(|v| v.as_str()).unwrap_or("");
                        if tag.is_empty() {
                            continue;
                        }
                        let map_to = entry
                            .get("map_to")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        config.dialects.push(KnownDialect {
                            tag: tag.to_string(),
                            map_to,
                        });
                    }
                }
                config.include_counterexamples = config.prompt_level != PromptLevel::L1;
            }
        }

        if let Some(native) = tool_calling.get("native") {
            if native.get("reliability").and_then(|v| v.as_str()) == Some("full") {
                config.lenient_parsing = false;
            }
        }

        Self::new(config)
    }
}

/// Observed text-tool deviation categories (align with ai-protocol `format.yaml` / §1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextToolDeviation {
    StandardToolCall,
    ShellDialect,
    BashDialect,
    DsmlDialect,
}

impl TextToolDeviation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StandardToolCall => "standard_tool_call",
            Self::ShellDialect => "shell",
            Self::BashDialect => "bash",
            Self::DsmlDialect => "dsml",
        }
    }
}

/// Detect the first recognizable text-tool markup in LLM output (for live validation logging).
pub fn detect_text_tool_deviation(text: &str) -> Option<TextToolDeviation> {
    if text.contains(DSML_TAG) {
        return Some(TextToolDeviation::DsmlDialect);
    }
    if shell_dialect_re().is_match(text) || shell_plain_body_re().is_match(text) {
        return Some(TextToolDeviation::ShellDialect);
    }
    if bash_dialect_re().is_match(text) {
        return Some(TextToolDeviation::BashDialect);
    }
    if tool_call_block_re().is_match(text) {
        return Some(TextToolDeviation::StandardToolCall);
    }
    None
}

/// Merge native structured tool calls with lenient text parsing when native is empty.
///
/// Runtime policy (ARCH-001): dialect parsing lives in ai-lib-core; applications only
/// wire dispatchers and must not implement provider-specific markup parsers.
pub fn parse_hybrid_tool_calls(
    parser: &impl TextToolParser,
    content: &str,
    native_calls: &[ToolCall],
) -> (String, Vec<ToolCall>) {
    if !native_calls.is_empty() {
        return (content.to_string(), native_calls.to_vec());
    }
    parser.parse(content)
}

impl TextToolParser for StandardTextToolParser {
    fn parse(&self, response_text: &str) -> (String, Vec<ToolCall>) {
        parse_text_tool_calls(response_text, &self.config)
    }

    fn prompt_instructions(&self, tools: &[ToolDefinition]) -> String {
        generate_prompt_instructions(tools, &self.config)
    }

    fn format_results(&self, results: &[ToolResult]) -> String {
        results
            .iter()
            .map(|r| {
                let body = serde_json::json!({
                    "tool_use_id": r.tool_use_id,
                    "content": r.content,
                    "is_error": r.is_error,
                });
                format!("<tool_result>\n{}\n</tool_result>", body)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn tool_call_block_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<tool_call(?:\s+[^>]*)?>(.*?)</tool_call>").expect("valid tool_call regex")
    })
}

fn shell_dialect_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<shell>\s*<command>(.*?)</command>\s*</shell>")
            .expect("valid shell dialect regex")
    })
}

fn shell_plain_body_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<shell>\s*(.*?)\s*</shell>").expect("valid shell plain body regex")
    })
}

fn bash_dialect_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<bash>(.*?)</bash>").expect("valid bash dialect regex"))
}

/// DeepSeek DSML delimiter: `<` + `｜｜DSML｜｜` (U+FF5C fullwidth vertical line).
const DSML_TAG: &str = "\u{FF5C}\u{FF5C}DSML\u{FF5C}\u{FF5C}";

fn dsml_invoke_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r#"(?s)<{DSML_TAG}invoke\s+name="([^"]+)">(.*?)</{DSML_TAG}invoke>"#
        ))
        .expect("valid dsml invoke regex")
    })
}

fn dsml_parameter_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r#"(?s)<{DSML_TAG}parameter\s+name="([^"]+)"[^>]*>(.*?)</{DSML_TAG}parameter>"#
        ))
        .expect("valid dsml parameter regex")
    })
}

/// Merge overlapping/adjacent byte spans for markup removal.
fn merge_spans(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if spans.is_empty() {
        return spans;
    }
    spans.sort_by_key(|(start, _)| *start);
    let mut merged = vec![spans[0]];
    for (start, end) in spans.into_iter().skip(1) {
        let last = merged.last_mut().expect("merged non-empty");
        if start <= last.1 {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// Parse DeepSeek DSML text tool calls (`<｜｜DSML｜｜invoke>` / `<｜｜DSML｜｜parameter>`).
fn parse_dsml_dialect(text: &str) -> (Vec<ToolCall>, Vec<(usize, usize)>) {
    let mut tool_calls = Vec::new();
    let mut spans_to_remove = Vec::new();
    let param_re = dsml_parameter_re();

    for caps in dsml_invoke_re().captures_iter(text) {
        let full = caps.get(0).unwrap();
        let tool_name = caps
            .get(1)
            .map(|m| m.as_str().trim())
            .unwrap_or("")
            .to_string();
        if tool_name.is_empty() {
            continue;
        }

        let body = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let mut arguments = serde_json::Map::new();
        for param_caps in param_re.captures_iter(body) {
            let key = param_caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let value = param_caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if !key.is_empty() {
                arguments.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }

        let idx = tool_calls.len();
        tool_calls.push(ToolCall {
            id: format!("text_tool_{idx}"),
            name: tool_name,
            arguments: serde_json::Value::Object(arguments),
        });
        spans_to_remove.push((full.start(), full.end()));
    }

    // Remove every DSML wrapper block (models may emit multiple per turn).
    let wrapper_re = Regex::new(&format!(
        r"(?s)<{DSML_TAG}tool_calls>\s*(.*?)\s*</{DSML_TAG}tool_calls>"
    ))
    .expect("valid dsml tool_calls wrapper regex");
    for caps in wrapper_re.captures_iter(text) {
        if let Some(full) = caps.get(0) {
            spans_to_remove.push((full.start(), full.end()));
        }
    }

    spans_to_remove = merge_spans(spans_to_remove);

    (tool_calls, spans_to_remove)
}

fn unwrap_tool_calls_wrapper(text: &str) -> String {
    let outer_re = Regex::new(r"(?s)<tool_calls>\s*(.*?)\s*</tool_calls>").unwrap();
    if let Some(caps) = outer_re.captures(text) {
        caps.get(1)
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| text.to_string())
    } else {
        text.to_string()
    }
}

fn extract_name_from_open_tag(full_match: &str) -> Option<String> {
    let attr_re = Regex::new(r#"name="([^"]+)""#).unwrap();
    attr_re
        .captures(full_match)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

fn normalize_arguments(
    obj: &serde_json::Map<String, serde_json::Value>,
    preferred_key: Option<&str>,
) -> serde_json::Value {
    if let Some(key) = preferred_key {
        if obj.contains_key(key) {
            return obj.get(key).cloned().unwrap_or(serde_json::json!({}));
        }
    }
    if obj.contains_key("arguments") {
        return obj
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));
    }
    for key in ["parameters", "params", "args"] {
        if let Some(v) = obj.get(key) {
            return v.clone();
        }
    }
    // Body is the arguments object itself (no wrapper keys).
    let mut args = obj.clone();
    args.remove("name");
    args.remove("id");
    args.remove("type");
    serde_json::Value::Object(args)
}

fn parse_json_body(
    body: &str,
    attr_name: Option<String>,
    preferred_args_key: Option<&str>,
) -> Option<(String, serde_json::Value)> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let obj = value.as_object()?;

    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or(attr_name)?;

    let arguments = normalize_arguments(obj, preferred_args_key);
    Some((name, arguments))
}

fn shell_tool_call(command: &str, map_to: &str, id: usize) -> ToolCall {
    let name = if map_to.is_empty() {
        "shell".to_string()
    } else {
        map_to.to_string()
    };
    ToolCall {
        id: format!("text_tool_{id}"),
        name,
        arguments: serde_json::json!({ "command": command }),
    }
}

fn try_parse_configured_dialects(
    text: &str,
    dialects: &[KnownDialect],
) -> Option<(ToolCall, (usize, usize))> {
    for dialect in dialects {
        match dialect.tag.as_str() {
            "shell" => {
                if let Some(caps) = shell_dialect_re().captures(text) {
                    let cmd = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                    let full = caps.get(0)?;
                    return Some((
                        shell_tool_call(cmd, &dialect.map_to, 0),
                        (full.start(), full.end()),
                    ));
                }
                if let Some(caps) = shell_plain_body_re().captures(text) {
                    let body = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                    if body.starts_with("<command>") {
                        continue;
                    }
                    let full = caps.get(0)?;
                    return Some((
                        shell_tool_call(body, &dialect.map_to, 0),
                        (full.start(), full.end()),
                    ));
                }
            }
            "bash" => {
                if let Some(caps) = bash_dialect_re().captures(text) {
                    let cmd = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                    let full = caps.get(0)?;
                    return Some((
                        shell_tool_call(cmd, &dialect.map_to, 0),
                        (full.start(), full.end()),
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

fn try_parse_legacy_dialects(text: &str) -> Option<(ToolCall, (usize, usize))> {
    if let Some(caps) = shell_dialect_re().captures(text) {
        let cmd = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let full = caps.get(0)?;
        return Some((shell_tool_call(cmd, "shell", 0), (full.start(), full.end())));
    }
    if let Some(caps) = shell_plain_body_re().captures(text) {
        let body = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if !body.starts_with("<command>") {
            let full = caps.get(0)?;
            return Some((
                shell_tool_call(body, "shell", 0),
                (full.start(), full.end()),
            ));
        }
    }
    if let Some(caps) = bash_dialect_re().captures(text) {
        let cmd = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let full = caps.get(0)?;
        return Some((shell_tool_call(cmd, "shell", 0), (full.start(), full.end())));
    }
    None
}

fn parse_text_tool_calls(text: &str, config: &TextToolConfig) -> (String, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut remaining = text.to_string();

    // L3: unwrap <tool_calls> wrapper when lenient
    if config.lenient_parsing {
        remaining = unwrap_tool_calls_wrapper(&remaining);
    }

    // Collect standard <tool_call> blocks
    let block_re = tool_call_block_re();
    let mut spans_to_remove: Vec<(usize, usize)> = Vec::new();

    for caps in block_re.captures_iter(&remaining) {
        let full = caps.get(0).unwrap();
        let body = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let attr_name = if config.lenient_parsing {
            extract_name_from_open_tag(full.as_str())
        } else {
            None
        };

        if let Some((name, arguments)) =
            parse_json_body(body, attr_name, config.args_key.as_deref())
        {
            let idx = tool_calls.len();
            tool_calls.push(ToolCall {
                id: format!("text_tool_{idx}"),
                name,
                arguments,
            });
            spans_to_remove.push((full.start(), full.end()));
        }
    }

    // L4: dialect adaptation when lenient and no standard blocks found
    if config.lenient_parsing && tool_calls.is_empty() {
        let (dsml_calls, dsml_spans) = parse_dsml_dialect(&remaining);
        if !dsml_calls.is_empty() {
            tool_calls.extend(dsml_calls);
            spans_to_remove.extend(dsml_spans);
        } else if let Some((call, span)) = if !config.dialects.is_empty() {
            try_parse_configured_dialects(&remaining, &config.dialects)
        } else {
            try_parse_legacy_dialects(&remaining)
        } {
            tool_calls.push(call);
            spans_to_remove.push(span);
        }
    }

    // Remove matched spans from remaining text (reverse order to preserve indices)
    spans_to_remove.sort_by_key(|(s, _)| *s);
    spans_to_remove.reverse();
    for (start, end) in spans_to_remove {
        if start <= remaining.len() && end <= remaining.len() {
            remaining.replace_range(start..end, "");
        }
    }

    let remaining_text = remaining
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    (remaining_text, tool_calls)
}

fn generate_prompt_instructions(tools: &[ToolDefinition], config: &TextToolConfig) -> String {
    let tool_list = tools
        .iter()
        .map(|t| {
            format!(
                "- {}: {}",
                t.function.name,
                t.function.description.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let is_zh = config.locale.starts_with("zh");

    match (config.prompt_level, is_zh) {
        (PromptLevel::L1, true) => format!(
            "## 工具调用协议\n\n\
             <tool_call>\n{{\"name\": \"工具名\", \"arguments\": {{\"参数\": \"值\"}}}}\n</tool_call>\n\n\
             可用工具：\n{tool_list}"
        ),
        (PromptLevel::L1, false) => format!(
            "## Tool Use Protocol\n\n\
             <tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{\"param\": \"value\"}}}}\n</tool_call>\n\n\
             Available tools:\n{tool_list}"
        ),
        (PromptLevel::L2, true) => format!(
            "## 工具调用协议\n\n\
             <tool_call>\n{{\"name\": \"工具名\", \"arguments\": {{\"参数\": \"值\"}}}}\n</tool_call>\n\n\
             关键规则：\n\
             - 只能使用 <tool_call>。<shell>、<bash>、<function> 将被忽略。\n\
             - JSON 必须包含 \"name\" 和 \"arguments\"。\n\n\
             可用工具：\n{tool_list}"
        ),
        (PromptLevel::L2, false) => format!(
            "## Tool Use Protocol\n\n\
             <tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{\"param\": \"value\"}}}}\n</tool_call>\n\n\
             CRITICAL RULES:\n\
             - Use <tool_call> ONLY. <shell>, <bash>, <function> WILL BE IGNORED.\n\
             - JSON must contain \"name\" (string) and \"arguments\" (object).\n\
             - Do NOT wrap in <tool_calls> or any other tag.\n\n\
             Available tools:\n{tool_list}"
        ),
        (PromptLevel::L3, _) => format!(
            "## Tool Use Protocol — Example\n\n\
             <tool_call>\n{{\"name\": \"shell\", \"arguments\": {{\"command\": \"ls -la\"}}}}\n</tool_call>\n\n\
             CRITICAL: <shell>, <bash>, <function> formats WILL BE IGNORED.\n\n\
             Available tools:\n{tool_list}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tool::FunctionDefinition;

    fn parser_strict() -> StandardTextToolParser {
        StandardTextToolParser::new(TextToolConfig {
            lenient_parsing: false,
            ..Default::default()
        })
    }

    fn parser_lenient() -> StandardTextToolParser {
        StandardTextToolParser::new(TextToolConfig {
            lenient_parsing: true,
            ..Default::default()
        })
    }

    #[test]
    fn strict_parse_standard_format() {
        let text = "I'll list the files for you.\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>";
        let (remaining, calls) = parser_strict().parse(text);
        assert_eq!(remaining, "I'll list the files for you.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls -la");
    }

    #[test]
    fn lenient_attribute_name() {
        let text = r#"<tool_call name="shell">{"command": "ls"}</tool_call>"#;
        let (_, calls) = parser_lenient().parse(text);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn lenient_nested_wrapper() {
        let text = r#"<tool_calls><tool_call id="1">{"name": "shell", "parameters": {"command": "ls"}}</tool_call></tool_calls>"#;
        let (_, calls) = parser_lenient().parse(text);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn lenient_field_alias() {
        let text = r#"<tool_call>{"name": "search", "params": {"query": "AI protocol", "limit": 10}}</tool_call>"#;
        let (_, calls) = parser_lenient().parse(text);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments["query"], "AI protocol");
    }

    #[test]
    fn lenient_shell_dialect() {
        let text = "Running command:\n<shell><command>ls</command></shell>";
        let (remaining, calls) = parser_lenient().parse(text);
        assert_eq!(remaining, "Running command:");
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn lenient_deepseek_dsml_dialect() {
        let tag = DSML_TAG;
        let text = format!(
            "我来检查 piubt 服务器上 pifan 服务的概况。\n\n\
             <{tag}tool_calls>\n\
             <{tag}invoke name=\"shell\">\n\
             <{tag}parameter name=\"command\" string=\"true\">ssh piubt \"systemctl status pifan\" 2>&1</{tag}parameter>\n\
             </{tag}invoke>\n\
             </{tag}tool_calls>"
        );
        let (remaining, calls) = parser_lenient().parse(&text);
        assert_eq!(remaining, "我来检查 piubt 服务器上 pifan 服务的概况。");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments["command"],
            "ssh piubt \"systemctl status pifan\" 2>&1"
        );
    }

    #[test]
    fn hybrid_prefers_native_calls() {
        let parser = parser_lenient();
        let native = vec![ToolCall {
            id: "call_1".to_string(),
            name: "shell".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        }];
        let text = "<shell><command>ignored</command></shell>";
        let (_, calls) = parse_hybrid_tool_calls(&parser, text, &native);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn lenient_deepseek_dsml_multiple_blocks() {
        let tag = DSML_TAG;
        let text = format!(
            "intro\n\
             <{tag}tool_calls>\n\
             <{tag}invoke name=\"file_read\">\n\
             <{tag}parameter name=\"file_path\" string=\"true\">/tmp/a</{tag}parameter>\n\
             </{tag}invoke>\n\
             </{tag}tool_calls>\n\
             middle\n\
             <{tag}tool_calls>\n\
             <{tag}invoke name=\"shell\">\n\
             <{tag}parameter name=\"command\" string=\"true\">grep foo</{tag}parameter>\n\
             </{tag}invoke>\n\
             </{tag}tool_calls>\n\
             outro"
        );
        let (remaining, calls) = parser_lenient().parse(&text);
        assert_eq!(remaining, "intro\nmiddle\noutro");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[1].name, "shell");
    }

    #[test]
    fn hybrid_falls_back_to_text_when_native_empty() {
        let parser = parser_lenient();
        let tag = DSML_TAG;
        let text = format!(
            "<{tag}invoke name=\"shell\">\
             <{tag}parameter name=\"command\">pwd</{tag}parameter>\
             </{tag}invoke>"
        );
        let (_, calls) = parse_hybrid_tool_calls(&parser, &text, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn detect_dsml_deviation() {
        let tag = DSML_TAG;
        let text = format!("<{tag}invoke name=\"shell\"></{tag}invoke>");
        assert_eq!(
            detect_text_tool_deviation(&text),
            Some(TextToolDeviation::DsmlDialect)
        );
    }

    #[test]
    fn prompt_l2_contains_counterexamples() {
        let tools = vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "shell".to_string(),
                description: Some("Execute shell commands".to_string()),
                parameters: None,
            },
        }];
        let parser = StandardTextToolParser::new(TextToolConfig {
            prompt_level: PromptLevel::L2,
            locale: "en".to_string(),
            ..Default::default()
        });
        let prompt = parser.prompt_instructions(&tools);
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("WILL BE IGNORED"));
        assert!(prompt.contains("shell"));
    }

    #[test]
    fn lenient_plain_shell_body_dialect() {
        let text =
            "让我检查一下。\n<shell>\nwhich opencode 2>/dev/null || echo \"not found\"\n</shell>";
        let parser = StandardTextToolParser::from_manifest_tool_calling(&serde_json::json!({
            "native": { "supported": true, "reliability": "partial" },
            "text_fallback": {
                "prompt_level": "L2",
                "known_dialects": [{ "tag": "shell", "map_to": "shell" }]
            }
        }));
        let (remaining, calls) = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments["command"],
            "which opencode 2>/dev/null || echo \"not found\""
        );
        assert!(remaining.contains("让我检查一下"));
    }

    #[test]
    fn tool_calling_policy_deepseek_partial_is_hybrid() {
        let tc = serde_json::json!({
            "native": { "supported": true, "reliability": "partial" },
            "text_fallback": { "prompt_level": "L2", "known_dialects": [{ "tag": "shell", "map_to": "shell" }] }
        });
        let policy = ToolCallingPolicy::from_tool_calling(Some(&tc));
        assert_eq!(policy.native_strategy, NativeStrategy::Hybrid);
        assert!(policy.send_native_tool_specs());
        assert!(policy.prefer_native_dispatcher());
    }

    #[test]
    fn tool_calling_policy_text_only_when_no_native() {
        let tc = serde_json::json!({
            "native": { "supported": false },
            "text_fallback": { "prompt_level": "L2" }
        });
        let policy = ToolCallingPolicy::from_tool_calling(Some(&tc));
        assert_eq!(policy.native_strategy, NativeStrategy::TextOnly);
        assert!(!policy.send_native_tool_specs());
    }
}
