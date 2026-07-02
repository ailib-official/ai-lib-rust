//! ALR-TTC-003 Phase 4 — live provider validation harness (DeepSeek).
//!
//! 文本工具调用 Phase 4 真机验证；`#[ignore]` + CI `--ignored` 时无密钥则 skip。

use ai_lib_core::types::text_tool::{
    PromptLevel, StandardTextToolParser, TextToolConfig, TextToolParser,
};
use ai_lib_core::types::tool::{FunctionDefinition, ToolDefinition};
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
                "provider": "deepseek",
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
