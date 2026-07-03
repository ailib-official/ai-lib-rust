//! ALR-TTC-003 Phase 4 — live provider validation harness (DeepSeek + Claude + Ollama).
//!
//! 文本工具调用 Phase 4 真机验证；`#[ignore]` + CI `--ignored` 时无密钥则 skip。

use ai_lib_core::types::text_tool::{
    detect_text_tool_deviation, parse_hybrid_tool_calls, PromptLevel, StandardTextToolParser,
    TextToolConfig, TextToolParser,
};
use ai_lib_core::types::tool::{FunctionDefinition, ToolCall, ToolDefinition};
use serde_json::{json, Value};
use std::io::Write;

const API_URL: &str = "https://api.deepseek.com/chat/completions";

struct Phase4Case {
    id: &'static str,
    user_msg: &'static str,
    expected_tools: &'static [&'static str],
    locale: &'static str,
}

const CASES: [Phase4Case; 5] = [
    Phase4Case {
        id: "P4-01",
        user_msg: "List files in current directory",
        expected_tools: &["shell"],
        locale: "en",
    },
    Phase4Case {
        id: "P4-02",
        user_msg: "Read README.md",
        expected_tools: &["file_read"],
        locale: "en",
    },
    Phase4Case {
        id: "P4-03",
        user_msg: "Run uname -a",
        expected_tools: &["shell"],
        locale: "en",
    },
    Phase4Case {
        id: "P4-04",
        user_msg: "列出当前目录文件",
        expected_tools: &["shell"],
        locale: "zh",
    },
    Phase4Case {
        id: "P4-05",
        user_msg: "read package.json and list dir",
        expected_tools: &["shell", "file_read"],
        locale: "en",
    },
];

fn phase4_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "shell".to_string(),
                description: Some("Run a shell command".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Shell command" }
                    },
                    "required": ["command"]
                })),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "file_read".to_string(),
                description: Some("Read a file path".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                })),
            },
        },
    ]
}

fn parser_for_locale(locale: &str) -> StandardTextToolParser {
    StandardTextToolParser::new(TextToolConfig {
        lenient_parsing: true,
        prompt_level: PromptLevel::L2,
        include_counterexamples: true,
        locale: locale.to_string(),
        args_key: Some("arguments".to_string()),
        ..Default::default()
    })
}

fn deepseek_api_key() -> Option<String> {
    std::env::var("DEEPSEEK_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn extract_message_content(json: &Value) -> String {
    let msg = &json["choices"][0]["message"];
    msg["content"]
        .as_str()
        .or_else(|| msg["reasoning_content"].as_str())
        .unwrap_or("")
        .to_string()
}

async fn deepseek_completion(model: &str, system: &str, user: &str) -> String {
    let key = deepseek_api_key().expect("caller must gate on key");
    let body = json!({
        "model": model,
        "temperature": 0,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
    });
    let client = reqwest::Client::new();
    let resp = client
        .post(API_URL)
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .expect("deepseek request");
    let status = resp.status();
    let text = resp.text().await.expect("response body");
    assert!(status.is_success(), "deepseek HTTP {status}: {text}");
    let json: Value = serde_json::from_str(&text).expect("json");
    extract_message_content(&json)
}

/// Mode B (ALR-TTC-003-R2b): native `tools` API — models may emit DSML/text markup in content.
async fn deepseek_completion_with_tools(model: &str, user: &str, tools: &[ToolDefinition]) -> Value {
    let key = deepseek_api_key().expect("caller must gate on key");
    let body = json!({
        "model": model,
        "temperature": 0,
        "messages": [{ "role": "user", "content": user }],
        "tools": tools,
        "tool_choice": "auto"
    });
    let client = reqwest::Client::new();
    let resp = client
        .post(API_URL)
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .expect("deepseek request");
    let status = resp.status();
    let text = resp.text().await.expect("response body");
    assert!(status.is_success(), "deepseek HTTP {status}: {text}");
    serde_json::from_str(&text).expect("json")
}

fn extract_native_tool_calls(msg: &Value) -> Vec<ToolCall> {
    msg["tool_calls"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    Some(ToolCall {
                        id: tc["id"].as_str()?.to_string(),
                        name: tc["function"]["name"].as_str()?.to_string(),
                        arguments: serde_json::from_str(tc["function"]["arguments"].as_str()?)
                            .unwrap_or(json!({})),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn case_matches(case: &Phase4Case, calls: &[ai_lib_core::types::tool::ToolCall]) -> bool {
    case.expected_tools
        .iter()
        .any(|name| calls.iter().any(|c| c.name == *name))
}

async fn run_rounds(model: &str, case: &Phase4Case, rounds: u32, min_success: u32) {
    let parser = parser_for_locale(case.locale);
    let tools = phase4_tools();
    let system = parser.prompt_instructions(&tools);
    let mut successes = 0u32;
    for round in 1..=rounds {
        let raw = deepseek_completion(model, &system, case.user_msg).await;
        let (_remainder, calls) = parser.parse(&raw);
        let ok = case_matches(case, &calls);
        if ok {
            successes += 1;
        }
        eprintln!(
            "{}",
            json!({
                "schema_version": 1,
                "task": "ALR-TTC-003",
                "call_mode": "text_prompt",
                "provider": "deepseek",
                "model": model,
                "round": round,
                "case_id": case.id,
                "prompt_lang": case.locale,
                "deviation": detect_text_tool_deviation(&raw).map(|d| d.as_str()),
                "success": ok,
                "tool_count": calls.len(),
            })
        );
    }
    assert!(
        successes >= min_success,
        "expected ≥{min_success}/{rounds} for {} on {model}, got {successes}",
        case.id
    );
}

fn skip_without_key(test_name: &str) -> bool {
    if deepseek_api_key().is_some() {
        return false;
    }
    eprintln!("{test_name}: DEEPSEEK_API_KEY not set, skipping");
    true
}

/// P4-01 × 5 rounds on deepseek-chat.
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_chat_p4_01_five_rounds() {
    if skip_without_key("deepseek_chat_p4_01_five_rounds") {
        return;
    }
    run_rounds("deepseek-chat", &CASES[0], 5, 3).await;
}

/// All P4 cases × 1 round on deepseek-chat.
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_chat_all_cases_one_round() {
    if skip_without_key("deepseek_chat_all_cases_one_round") {
        return;
    }
    for case in &CASES {
        run_rounds("deepseek-chat", case, 1, 1).await;
    }
}

/// P4-01 × 5 rounds on deepseek-reasoner.
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_reasoner_p4_01_five_rounds() {
    if skip_without_key("deepseek_reasoner_p4_01_five_rounds") {
        return;
    }
    run_rounds("deepseek-reasoner", &CASES[0], 5, 2).await;
}

/// R2b — Mode B: native `tools` API + `parse_hybrid_tool_calls` (production-like path).
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_chat_p4_01_native_tools_hybrid() {
    if skip_without_key("deepseek_chat_p4_01_native_tools_hybrid") {
        return;
    }
    let case = &CASES[0];
    let tools = phase4_tools();
    let parser = parser_for_locale(case.locale);
    let model = "deepseek-chat";
    let mut successes = 0u32;
    let rounds = 3;
    for round in 1..=rounds {
        let resp = deepseek_completion_with_tools(model, case.user_msg, &tools).await;
        let msg = &resp["choices"][0]["message"];
        let content = extract_message_content(&resp);
        let native = extract_native_tool_calls(msg);
        let (_remainder, calls) = parse_hybrid_tool_calls(&parser, &content, &native);
        let ok = case_matches(case, &calls);
        if ok {
            successes += 1;
        }
        eprintln!(
            "{}",
            json!({
                "schema_version": 1,
                "task": "ALR-TTC-003-R2b",
                "call_mode": "native_tools",
                "provider": "deepseek",
                "model": model,
                "round": round,
                "case_id": case.id,
                "native_tool_count": native.len(),
                "deviation": detect_text_tool_deviation(&content).map(|d| d.as_str()),
                "success": ok,
                "tool_count": calls.len(),
            })
        );
    }
    assert!(
        successes >= 1,
        "R2b: expected ≥1/{rounds} hybrid parse success on {model}, got {successes}"
    );
}

/// Optional JSONL export when `TTC_PHASE4_OUT` is set.
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_chat_jsonl_export() {
    let out = match std::env::var("TTC_PHASE4_OUT") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    if deepseek_api_key().is_none() {
        eprintln!("deepseek_chat_jsonl_export: DEEPSEEK_API_KEY not set, skipping");
        return;
    }

    let model = "deepseek-chat";
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .expect("open TTC_PHASE4_OUT");

    for case in &CASES {
        let parser = parser_for_locale(case.locale);
        let system = parser.prompt_instructions(&phase4_tools());
        for round in 1..=5 {
            let raw = deepseek_completion(model, &system, case.user_msg).await;
            let (_remainder, calls) = parser.parse(&raw);
            let ok = case_matches(case, &calls);
            let record = json!({
                "schema_version": 1,
                "task": "ALR-TTC-003",
                "provider": "deepseek",
                "model": model,
                "round": round,
                "case_id": case.id,
                "prompt_lang": case.locale,
                "raw_output": raw,
                "success": ok,
                "tool_count": calls.len(),
            });
            writeln!(file, "{record}").expect("write jsonl");
        }
    }
}

// --- Anthropic (ALR-TTC-003-R3) ---

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";

fn anthropic_api_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn skip_without_anthropic_key(test_name: &str) -> bool {
    if anthropic_api_key().is_some() {
        return false;
    }
    eprintln!("{test_name}: ANTHROPIC_API_KEY not set, skipping");
    true
}

fn extract_anthropic_text(json: &Value) -> String {
    json["content"]
        .as_array()
        .and_then(|blocks| blocks.first())
        .and_then(|b| b["text"].as_str())
        .unwrap_or("")
        .to_string()
}

async fn anthropic_completion(model: &str, system: &str, user: &str) -> String {
    let key = anthropic_api_key().expect("caller must gate on key");
    let body = json!({
        "model": model,
        "max_tokens": 1024,
        "temperature": 0,
        "system": system,
        "messages": [{ "role": "user", "content": user }]
    });
    let client = reqwest::Client::new();
    let resp = client
        .post(ANTHROPIC_URL)
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .expect("anthropic request");
    let status = resp.status();
    let text = resp.text().await.expect("response body");
    assert!(status.is_success(), "anthropic HTTP {status}: {text}");
    let json: Value = serde_json::from_str(&text).expect("json");
    extract_anthropic_text(&json)
}

async fn run_claude_rounds(model: &str, case: &Phase4Case, rounds: u32, min_success: u32) {
    let parser = parser_for_locale(case.locale);
    let tools = phase4_tools();
    let system = parser.prompt_instructions(&tools);
    let mut successes = 0u32;
    for round in 1..=rounds {
        let raw = anthropic_completion(model, &system, case.user_msg).await;
        let (_remainder, calls) = parser.parse(&raw);
        let ok = case_matches(case, &calls);
        if ok {
            successes += 1;
        }
        eprintln!(
            "{}",
            json!({
                "schema_version": 1,
                "task": "ALR-TTC-003",
                "provider": "anthropic",
                "model": model,
                "round": round,
                "case_id": case.id,
                "prompt_lang": case.locale,
                "success": ok,
                "tool_count": calls.len(),
            })
        );
    }
    assert!(
        successes >= min_success,
        "expected ≥{min_success}/{rounds} for {} on {model}, got {successes}",
        case.id
    );
}

/// P4-01 × 5 on claude-sonnet-4-6 (text path, no native tools).
#[tokio::test]
#[ignore = "live Anthropic API — requires ANTHROPIC_API_KEY"]
async fn claude_sonnet_p4_01_five_rounds() {
    if skip_without_anthropic_key("claude_sonnet_p4_01_five_rounds") {
        return;
    }
    run_claude_rounds("claude-sonnet-4-6", &CASES[0], 5, 3).await;
}

/// P4-01 × 5 on claude-opus-4-8.
#[tokio::test]
#[ignore = "live Anthropic API — requires ANTHROPIC_API_KEY"]
async fn claude_opus_p4_01_five_rounds() {
    if skip_without_anthropic_key("claude_opus_p4_01_five_rounds") {
        return;
    }
    run_claude_rounds("claude-opus-4-8", &CASES[0], 5, 2).await;
}

/// All P4 cases × 1 round on claude-sonnet-4-6.
#[tokio::test]
#[ignore = "live Anthropic API — requires ANTHROPIC_API_KEY"]
async fn claude_sonnet_all_cases_one_round() {
    if skip_without_anthropic_key("claude_sonnet_all_cases_one_round") {
        return;
    }
    for case in &CASES {
        run_claude_rounds("claude-sonnet-4-6", case, 1, 1).await;
    }
}

// --- Ollama local (ALR-TTC-003-R4) ---

fn ollama_host() -> String {
    std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
        .trim()
        .trim_end_matches('/')
        .to_string()
}

async fn ollama_reachable() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(format!("{}/api/tags", ollama_host()))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn ollama_model_available(model: &str) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let Ok(resp) = client
        .get(format!("{}/api/tags", ollama_host()))
        .send()
        .await
    else {
        return false;
    };
    let Ok(text) = resp.text().await else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    json["models"].as_array().is_some_and(|models| {
        models.iter().any(|m| {
            m["name"]
                .as_str()
                .is_some_and(|name| name == model || name.starts_with(&format!("{model}:")))
        })
    })
}

async fn skip_without_ollama(test_name: &str, model: &str) -> bool {
    if !ollama_reachable().await {
        eprintln!(
            "{test_name}: Ollama not reachable at {}, skipping",
            ollama_host()
        );
        return true;
    }
    if !ollama_model_available(model).await {
        eprintln!("{test_name}: model {model} not in Ollama tags, skipping");
        return true;
    }
    false
}

fn extract_ollama_content(json: &Value) -> String {
    json["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

async fn ollama_completion(model: &str, system: &str, user: &str) -> String {
    let body = json!({
        "model": model,
        "stream": false,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ],
        "options": { "temperature": 0 }
    });
    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", ollama_host());
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .expect("ollama request");
    let status = resp.status();
    let text = resp.text().await.expect("response body");
    assert!(status.is_success(), "ollama HTTP {status}: {text}");
    let json: Value = serde_json::from_str(&text).expect("json");
    extract_ollama_content(&json)
}

async fn run_ollama_rounds(model: &str, case: &Phase4Case, rounds: u32, min_success: u32) {
    let parser = parser_for_locale(case.locale);
    let tools = phase4_tools();
    let system = parser.prompt_instructions(&tools);
    let mut successes = 0u32;
    for round in 1..=rounds {
        let raw = ollama_completion(model, &system, case.user_msg).await;
        let (_remainder, calls) = parser.parse(&raw);
        let ok = case_matches(case, &calls);
        if ok {
            successes += 1;
        }
        eprintln!(
            "{}",
            json!({
                "schema_version": 1,
                "task": "ALR-TTC-003-R4",
                "call_mode": "text_prompt",
                "provider": "ollama",
                "model": model,
                "round": round,
                "case_id": case.id,
                "prompt_lang": case.locale,
                "deviation": detect_text_tool_deviation(&raw).map(|d| d.as_str()),
                "success": ok,
                "tool_count": calls.len(),
            })
        );
    }
    assert!(
        successes >= min_success,
        "expected ≥{min_success}/{rounds} for {} on {model}, got {successes}",
        case.id
    );
}

/// P4-01 × 5 on llama3 (local).
#[tokio::test]
#[ignore = "live Ollama — requires local daemon + llama3 model"]
async fn ollama_llama3_p4_01_five_rounds() {
    const MODEL: &str = "llama3";
    if skip_without_ollama("ollama_llama3_p4_01_five_rounds", MODEL).await {
        return;
    }
    run_ollama_rounds(MODEL, &CASES[0], 5, 2).await;
}

/// P4-01 × 5 on qwen2.5 (local).
#[tokio::test]
#[ignore = "live Ollama — requires local daemon + qwen2.5 model"]
async fn ollama_qwen25_p4_01_five_rounds() {
    const MODEL: &str = "qwen2.5";
    if skip_without_ollama("ollama_qwen25_p4_01_five_rounds", MODEL).await {
        return;
    }
    run_ollama_rounds(MODEL, &CASES[0], 5, 2).await;
}

/// P4-01 × 5 on mistral (local).
#[tokio::test]
#[ignore = "live Ollama — requires local daemon + mistral model"]
async fn ollama_mistral_p4_01_five_rounds() {
    const MODEL: &str = "mistral";
    if skip_without_ollama("ollama_mistral_p4_01_five_rounds", MODEL).await {
        return;
    }
    run_ollama_rounds(MODEL, &CASES[0], 5, 2).await;
}

/// All P4 cases × 1 round on llama3.
#[tokio::test]
#[ignore = "live Ollama — requires local daemon + llama3 model"]
async fn ollama_llama3_all_cases_one_round() {
    const MODEL: &str = "llama3";
    if skip_without_ollama("ollama_llama3_all_cases_one_round", MODEL).await {
        return;
    }
    for case in &CASES {
        run_ollama_rounds(MODEL, case, 1, 1).await;
    }
}
