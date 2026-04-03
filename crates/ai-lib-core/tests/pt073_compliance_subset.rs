//! PT-073 / Wave-5: core-only compliance subset (protocol_loading + message_building).
//!
//! Mirrors the logic in `crates/ai-lib-rust/tests/compliance.rs` for these two test types
//! so `cargo test -p ai-lib-core` can validate the execution layer without the facade crate.
//! Full matrix (error_classification, streaming, etc.) remains on `ai-lib-rust --test compliance`.
//!
//! Set `COMPLIANCE_DIR` to the `ai-protocol/tests/compliance` directory if auto-discovery fails.

use serde::Deserialize;
use serde_yaml::Value;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct TestCase {
    suite: String,
    name: String,
    id: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    setup: Option<TestSetup>,
    input: TestInput,
    expected: TestExpected,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct TestSetup {
    provider: Option<String>,
    manifest_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TestInput {
    #[serde(rename = "type")]
    test_type: String,
    #[serde(default)]
    manifest_path: Option<String>,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
struct TestExpected {
    #[serde(default)]
    valid: Option<bool>,
    #[serde(default)]
    provider_id: Option<String>,
    #[serde(default)]
    protocol_version: Option<String>,
    #[serde(default)]
    errors: Option<Value>,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

fn compliance_dir() -> PathBuf {
    if let Ok(dir) = env::var("COMPLIANCE_DIR") {
        return PathBuf::from(dir);
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        // Core crate lives at crates/ai-lib-core — ai-protocol often sibling of workspace root
        manifest_dir.join("../../../ai-protocol/tests/compliance"),
        manifest_dir.join("../../../../ai-protocol/tests/compliance"),
        // Same layouts as facade crate
        manifest_dir.join("../../ai-protocol/tests/compliance"),
        manifest_dir.join("../ai-protocol/tests/compliance"),
    ];
    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }

    manifest_dir.join("../../../ai-protocol/tests/compliance")
}

fn discover_yaml_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() {
        return files;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(discover_yaml_files(&path));
            } else if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn parse_test_cases(content: &str) -> Vec<TestCase> {
    let content = content.replace("\r\n", "\n");
    let mut cases = Vec::new();
    for document in serde_yaml::Deserializer::from_str(&content) {
        match TestCase::deserialize(document) {
            Ok(tc) => cases.push(tc),
            Err(e) => eprintln!("  [WARN] Skipped non-test-case document: {}", e),
        }
    }
    cases
}

fn manifest_has_required_shape(manifest: &Value) -> bool {
    let id_ok = manifest
        .get("id")
        .and_then(Value::as_str)
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let pv_ok = manifest
        .get("protocol_version")
        .and_then(Value::as_str)
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let endpoint_ok = manifest
        .get("endpoint")
        .and_then(Value::as_mapping)
        .and_then(|m| m.get(Value::String("base_url".to_string())))
        .and_then(Value::as_str)
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    id_ok && pv_ok && endpoint_ok
}

fn capability_profile_phase_errors(manifest: &Value) -> Vec<String> {
    let Some(cp) = manifest.get("capability_profile") else {
        return Vec::new();
    };
    let Some(cp_map) = cp.as_mapping() else {
        return vec!["capability_profile must be object".to_string()];
    };

    let phase = cp_map
        .get(Value::String("phase".to_string()))
        .and_then(Value::as_str)
        .unwrap_or_default();

    let mut errors = Vec::new();
    let has_ios_keys = || {
        cp_map.contains_key(Value::String("inputs".to_string()))
            || cp_map.contains_key(Value::String("outcomes".to_string()))
            || cp_map.contains_key(Value::String("systems".to_string()))
    };

    match phase {
        "ios_v1" => {
            if cp_map.contains_key(Value::String("process".to_string()))
                || cp_map.contains_key(Value::String("contract".to_string()))
            {
                errors.push("must NOT have additional properties".to_string());
            }
            if !has_ios_keys() {
                errors.push("must match at least one schema in anyOf".to_string());
            }
        }
        "iospc_v1" => {
            if !has_ios_keys() {
                errors.push("iospc_v1 requires inputs or outcomes or systems".to_string());
            }
            if !cp_map.contains_key(Value::String("process".to_string()))
                && !cp_map.contains_key(Value::String("contract".to_string()))
            {
                errors.push("iospc_v1 requires process or contract".to_string());
            }
        }
        "" => {}
        _ => errors.push("phase must be ios_v1 or iospc_v1".to_string()),
    }

    errors
}

fn run_protocol_loading(tc: &TestCase, compliance_dir: &Path) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    let manifest_rel = tc
        .input
        .manifest_path
        .as_ref()
        .or_else(|| tc.setup.as_ref().and_then(|s| s.manifest_path.as_ref()));
    let Some(manifest_rel) = manifest_rel else {
        return Err(vec!["protocol_loading requires manifest_path".to_string()]);
    };

    let manifest_path = compliance_dir.join(manifest_rel);
    let raw = fs::read_to_string(&manifest_path).map_err(|e| {
        vec![format!(
            "failed to read manifest {}: {}",
            manifest_path.display(),
            e
        )]
    })?;
    let manifest: Value = serde_yaml::from_str(&raw).map_err(|e| {
        vec![format!(
            "failed to parse manifest {}: {}",
            manifest_path.display(),
            e
        )]
    })?;

    let cp_errors = capability_profile_phase_errors(&manifest);
    let actual_valid = manifest_has_required_shape(&manifest) && cp_errors.is_empty();
    let expected_valid = tc.expected.valid.unwrap_or(false);
    if actual_valid != expected_valid {
        failures.push(format!(
            "valid: expected {}, got {}",
            expected_valid, actual_valid
        ));
    }

    if let Some(expected_errors) = tc.expected.errors.as_ref().and_then(Value::as_sequence) {
        let actual_error_text = cp_errors.join(" | ");
        for expected in expected_errors {
            if let Some(expected_text) = expected.as_str() {
                if !actual_error_text.contains(expected_text) {
                    failures.push(format!(
                        "errors: expected '{}' not found in '{}'",
                        expected_text, actual_error_text
                    ));
                }
            }
        }
    }

    if expected_valid {
        if let Some(expected_provider_id) = tc.expected.provider_id.as_ref() {
            let got = manifest
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if got != expected_provider_id {
                failures.push(format!(
                    "provider_id: expected {}, got {}",
                    expected_provider_id, got
                ));
            }
        }
        if let Some(expected_protocol_version) = tc.expected.protocol_version.as_ref() {
            let got = manifest
                .get("protocol_version")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if got != expected_protocol_version {
                failures.push(format!(
                    "protocol_version: expected {}, got {}",
                    expected_protocol_version, got
                ));
            }
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn run_message_building(tc: &TestCase) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    let messages = tc
        .input
        .extra
        .get("messages")
        .and_then(Value::as_sequence)
        .cloned()
        .unwrap_or_default();
    let expected_messages = tc
        .expected
        .extra
        .get("normalized_body")
        .and_then(Value::as_mapping)
        .and_then(|m| m.get(Value::String("messages".to_string())))
        .and_then(Value::as_sequence)
        .cloned()
        .unwrap_or_default();
    if messages != expected_messages {
        failures.push("normalized messages mismatch".to_string());
    }
    let expected_count = tc
        .expected
        .extra
        .get("message_count")
        .and_then(Value::as_u64)
        .unwrap_or(expected_messages.len() as u64) as usize;
    if messages.len() != expected_count {
        failures.push(format!(
            "message_count: expected {}, got {}",
            expected_count,
            messages.len()
        ));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

#[test]
fn pt073_protocol_loading_cases() {
    let compliance_dir = compliance_dir();
    if !compliance_dir.exists() {
        eprintln!(
            "[SKIP] COMPLIANCE_DIR / ai-protocol not found: {}",
            compliance_dir.display()
        );
        return;
    }

    let loading_dir = compliance_dir.join("cases/01-protocol-loading");
    if !loading_dir.exists() {
        eprintln!(
            "[SKIP] Protocol loading cases dir missing: {}",
            loading_dir.display()
        );
        return;
    }

    let yaml_files = discover_yaml_files(&loading_dir);
    let mut passed = 0u32;
    let mut failed = 0u32;

    for file in yaml_files {
        let content = match fs::read_to_string(&file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  [WARN] Could not read {}: {}", file.display(), e);
                continue;
            }
        };

        for tc in parse_test_cases(&content) {
            if tc.input.test_type != "protocol_loading" {
                continue;
            }

            match run_protocol_loading(&tc, &compliance_dir) {
                Ok(()) => {
                    println!("  [PASS] {} ({})", tc.id, tc.name);
                    passed += 1;
                }
                Err(failures) => {
                    println!("  [FAIL] {} ({})", tc.id, tc.name);
                    for f in &failures {
                        println!("         {}", f);
                    }
                    failed += 1;
                }
            }
        }
    }

    println!("\n--- PT-073 core protocol_loading summary ---");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);

    assert_eq!(
        failed, 0,
        "{} protocol_loading compliance test(s) failed",
        failed
    );
}

#[test]
fn pt073_message_building_cases() {
    let compliance_dir = compliance_dir();
    if !compliance_dir.exists() {
        eprintln!(
            "[SKIP] COMPLIANCE_DIR / ai-protocol not found: {}",
            compliance_dir.display()
        );
        return;
    }

    let dir = compliance_dir.join("cases/03-message-building");
    if !dir.exists() {
        eprintln!(
            "[SKIP] Message building cases dir missing: {}",
            dir.display()
        );
        return;
    }

    let yaml_files = discover_yaml_files(&dir);
    let mut passed = 0u32;
    let mut failed = 0u32;

    for file in yaml_files {
        let content = match fs::read_to_string(&file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  [WARN] Could not read {}: {}", file.display(), e);
                continue;
            }
        };

        for tc in parse_test_cases(&content) {
            if tc.input.test_type != "message_building" {
                continue;
            }

            match run_message_building(&tc) {
                Ok(()) => {
                    println!("  [PASS] {} ({})", tc.id, tc.name);
                    passed += 1;
                }
                Err(failures) => {
                    println!("  [FAIL] {} ({})", tc.id, tc.name);
                    for f in &failures {
                        println!("         {}", f);
                    }
                    failed += 1;
                }
            }
        }
    }

    println!("\n--- PT-073 core message_building summary ---");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);

    assert_eq!(
        failed, 0,
        "{} message_building compliance test(s) failed",
        failed
    );
}
