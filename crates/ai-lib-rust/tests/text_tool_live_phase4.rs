//! ALR-TTC-003 Phase 4 — live provider validation harness (DeepSeek).
//!
//! 文本工具调用 Phase 4 真机验证；默认 `#[ignore]`，CI 不执行。

use ai_lib_core::types::text_tool::{
    PromptLevel, StandardTextToolParser, TextToolConfig, TextToolParser,
};
use ai_lib_core::types::tool::{FunctionDefinition, ToolDefinition};
use serde_json::json;
use std::io::Write;

const API_URL: &str = "https://api.deepseek.com/chat/completions";

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

fn deepseek_parser() -> StandardTextToolParser {
    StandardTextToolParser::new(TextToolConfig {
        lenient_parsing: true,
        prompt_level: PromptLevel::L2,
        include_counterexamples: true,
        locale: "en".to_string(),
        args_key: Some("arguments".to_string()),
        ..Default::default()
    })
}

async fn deepseek_completion(model: &str, system: &str, user: &str) -> String {
    let key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY");
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
    let json: serde_json::Value = serde_json::from_str(&text).expect("json");
    json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn case_success(tool_name: &str, calls: &[ai_lib_core::types::tool::ToolCall]) -> bool {
    calls.iter().any(|c| c.name == tool_name)
}

/// P4-01 × 5 rounds on deepseek-chat — ALR-TTC-003-R2 acceptance smoke.
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_chat_p4_01_five_rounds() {
    let parser = deepseek_parser();
    let tools = phase4_tools();
    let system = parser.prompt_instructions(&tools);
    let user_msg = "List files in current directory";
    let model = "deepseek-chat";

    let mut successes = 0u32;
    for round in 1..=5 {
        let raw = deepseek_completion(model, &system, user_msg).await;
        let (_remainder, calls) = parser.parse(&raw);
        let ok = case_success("shell", &calls);
        if ok {
            successes += 1;
        }
        let record = json!({
            "schema_version": 1,
            "task": "ALR-TTC-003",
            "provider": "deepseek",
            "model": model,
            "round": round,
            "case_id": "P4-01",
            "prompt_lang": "en",
            "success": ok,
            "tool_count": calls.len(),
        });
        eprintln!("{}", record);
    }
    assert!(
        successes >= 3,
        "expected ≥3/5 parse successes for P4-01 on {model}, got {successes}"
    );
}

/// Optional: write JSONL records to path from `TTC_PHASE4_OUT`.
#[tokio::test]
#[ignore = "live DeepSeek API — requires DEEPSEEK_API_KEY"]
async fn deepseek_chat_p4_01_jsonl_export() {
    let out = match std::env::var("TTC_PHASE4_OUT") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };

    let parser = deepseek_parser();
    let tools = phase4_tools();
    let system = parser.prompt_instructions(&tools);
    let model = "deepseek-chat";
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .expect("open TTC_PHASE4_OUT");

    for round in 1..=5 {
        let raw = deepseek_completion(model, &system, "List files in current directory").await;
        let (_remainder, calls) = parser.parse(&raw);
        let ok = case_success("shell", &calls);
        let record = json!({
            "schema_version": 1,
            "task": "ALR-TTC-003",
            "provider": "deepseek",
            "model": model,
            "round": round,
            "case_id": "P4-01",
            "prompt_lang": "en",
            "raw_output": raw,
            "success": ok,
            "tool_count": calls.len(),
        });
        writeln!(file, "{record}").expect("write jsonl");
    }
}
