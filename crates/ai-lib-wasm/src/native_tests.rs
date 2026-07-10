//! Native-target unit tests for ai-lib-wasm.
//!
//! Chinese: ai-lib-wasm native-target unit tests.

use std::sync::Mutex;

use crate::abi::*;
use crate::buffers::*;
use crate::invoke::*;
use crate::io::*;
use crate::memory::*;
use crate::snapshot_ops::*;
use crate::state::*;
use crate::v1::*;
use crate::AILIB_ABI_VERSION;

/// All state-mutating tests take this lock so they never race on the
/// module-global statics (`MANIFESTS`, `METRICS`, `LAST_OUT`, `LAST_ERR`).
/// `cargo test` default parallelism would otherwise interleave them.
fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    // Poisoning can happen if an earlier test panicked while holding the
    // lock; recover because subsequent tests can still run safely.
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn reset_state() {
    if let Ok(mut g) = MANIFESTS.lock() {
        g.clear();
    }
    if let Ok(mut g) = METRICS.lock() {
        *g = WasmMetrics::default();
    }
    ailib_arena_reset();
}

/// Minimal valid manifest YAML for `load_manifest_validated` (ProtocolManifest v2).
const MIN_MANIFEST_YAML: &str = r#"---
id: test-provider
name: Test Provider
version: "1.0.0"
protocol_version: "2.0"
endpoint:
  base_url: https://api.example.com/v1
capabilities:
  streaming: true
  tools: false
  vision: false
status: stable
category: ai_provider
official_url: https://example.com
support_contact: https://example.com/support
parameter_mappings: {}
"#;

const CREDENTIAL_MANIFEST_YAML: &str = r#"---
id: replicate
name: Replicate Credential Mock
version: "1.0.0"
protocol_version: "2.0"
endpoint:
  base_url: https://api.replicate.example/v1
  chat: /chat/completions
  auth:
    type: bearer
    token_env: REPLICATE_API_TOKEN
capabilities:
  streaming: false
  tools: false
  vision: false
status: stable
category: ai_provider
official_url: https://example.com
support_contact: https://example.com/support
parameter_mappings: {}
"#;

fn invoke(op: &str, input: Option<&[u8]>, ctx: Option<&[u8]>) -> i32 {
    let (ip, il) = input
        .map(|b| (b.as_ptr(), b.len()))
        .unwrap_or((std::ptr::null(), 0));
    let (cp, cl) = ctx
        .map(|b| (b.as_ptr(), b.len()))
        .unwrap_or((std::ptr::null(), 0));
    unsafe { ailib_invoke(op.as_ptr(), op.len(), ip, il, cp, cl) }
}

fn read_err() -> String {
    LAST_ERR
        .lock()
        .map(|g| String::from_utf8_lossy(&g).into_owned())
        .unwrap_or_default()
}

fn read_out() -> Vec<u8> {
    LAST_OUT.lock().map(|g| g.clone()).unwrap_or_default()
}

// ---- WASM-001: ABI version + capabilities ---------------------------

#[test]
fn test_ailib_abi_version() {
    assert_eq!(ailib_abi_version(), AILIB_ABI_VERSION);
    assert!(AILIB_ABI_VERSION >= 2, "WASM-001 requires v2+");
}

#[test]
fn test_ailib_capabilities_json_valid() {
    let ptr = ailib_capabilities_ptr();
    let len = ailib_capabilities_len();
    assert!(!ptr.is_null());
    assert!(len > 0);
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(v["version"], AILIB_ABI_VERSION);
    let ops = v["ops"].as_array().unwrap();
    let op_names: Vec<&str> = ops.iter().filter_map(|o| o.as_str()).collect();
    for must in &[
        "build_request",
        "parse_response",
        "classify_error",
        "resolve_credential",
        "load_manifest",
        "snapshot_state",
        "restore_state",
    ] {
        assert!(op_names.contains(must), "capabilities missing op {must}");
    }
}

#[test]
fn test_ailib_invoke_abi_version_op() {
    let _g = test_lock();
    reset_state();
    assert_eq!(invoke("abi_version", None, None), 0);
    let v: serde_json::Value = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(v["version"], AILIB_ABI_VERSION);
}

#[test]
fn test_ailib_invoke_unknown_op() {
    let _g = test_lock();
    reset_state();
    assert_eq!(invoke("nope", None, None), -1);
    assert!(read_err().contains("unknown op"));
}

#[test]
fn test_ailib_invoke_rejects_future_ctx_version() {
    let _g = test_lock();
    reset_state();
    let ctx = br#"{"version": 999}"#;
    assert_eq!(invoke("abi_version", None, Some(ctx)), -1);
    assert!(read_err().contains("newer than AILIB_ABI_VERSION"));
}

#[test]
fn test_ailib_invoke_v1_caller_against_v2() {
    // Cross-version: v1 host passes ctx.version=1 against v2 WASM. Must succeed.
    let _g = test_lock();
    reset_state();
    let ctx = br#"{"version": 1}"#;
    assert_eq!(invoke("capabilities", None, Some(ctx)), 0);
    let v: serde_json::Value = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(v["version"], AILIB_ABI_VERSION);
}

#[test]
fn test_ailib_invoke_classify_error() {
    let _g = test_lock();
    reset_state();
    let ctx = br#"{"version": 2, "status_code": 429}"#;
    assert_eq!(invoke("classify_error", None, Some(ctx)), 0);
    let v: serde_json::Value = serde_json::from_slice(&read_out()).unwrap();
    assert!(v["code"].as_str().unwrap().starts_with('E'));
}

#[test]
fn test_ailib_invoke_load_and_check_capability() {
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1, "load_manifest failed: {}", read_err());

    let input = format!(r#"{{"handle": {}, "name": "streaming"}}"#, h);
    let ctx = br#"{"version": 2}"#;
    assert_eq!(
        invoke("check_capability", Some(input.as_bytes()), Some(ctx)),
        0
    );
    let v: serde_json::Value = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(v["supported"], true);
}

#[test]
fn test_ailib_invoke_build_request_uses_ctx_handle() {
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1);
    let req = serde_json::json!({
        "model": "test-model",
        "messages": [{"role":"user","content":"hi"}],
        "stream": false
    });
    let req_bytes = serde_json::to_vec(&req).unwrap();
    let ctx = format!(r#"{{"version": 2, "manifest_handle": {}}}"#, h);
    assert_eq!(
        invoke("build_request", Some(&req_bytes), Some(ctx.as_bytes())),
        0,
        "err={}",
        read_err()
    );
    let out: serde_json::Value = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(out["model"], "test-model");
}

#[test]
fn test_ailib_invoke_metrics_op() {
    let _g = test_lock();
    reset_state();
    // Trigger at least one call so metrics move.
    unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert_eq!(invoke("metrics", None, None), 0);
    let m: WasmMetrics = serde_json::from_slice(&read_out()).unwrap();
    assert!(m.total_calls >= 1);
}

#[test]
fn test_ailib_invoke_resolve_credential_uses_host_supplied_only() {
    let _g = test_lock();
    reset_state();
    std::env::set_var("REPLICATE_API_TOKEN", "host-env-token-ignored-by-wasm");
    let h = unsafe {
        ailib_load_manifest(
            CREDENTIAL_MANIFEST_YAML.as_ptr(),
            CREDENTIAL_MANIFEST_YAML.len(),
        )
    };
    assert!(h >= 1, "load_manifest failed: {}", read_err());
    let input = br#"{"explicit_credential":"host-supplied-token"}"#;
    let ctx = format!(r#"{{"version": 2, "manifest_handle": {}}}"#, h);
    assert_eq!(
        invoke("resolve_credential", Some(input), Some(ctx.as_bytes())),
        0,
        "err={}",
        read_err()
    );
    let out_bytes = read_out();
    let out: serde_json::Value = serde_json::from_slice(&out_bytes).unwrap();
    assert_eq!(out["status"], "available");
    assert_eq!(out["source_kind"], "explicit");
    assert_eq!(out["source_name"], "host_supplied");
    assert_eq!(out["required"][0], "REPLICATE_API_TOKEN");
    assert_eq!(out["implicit_env_allowed"], false);
    assert_eq!(out["implicit_keyring_allowed"], false);
    let public = String::from_utf8(out_bytes).unwrap();
    assert!(!public.contains("host-supplied-token"));
    assert!(!public.contains("host-env-token-ignored-by-wasm"));
    std::env::remove_var("REPLICATE_API_TOKEN");
}

#[test]
fn test_ailib_invoke_resolve_credential_ignores_env_without_explicit() {
    let _g = test_lock();
    reset_state();
    std::env::set_var("REPLICATE_API_TOKEN", "host-env-token-ignored-by-wasm");
    let h = unsafe {
        ailib_load_manifest(
            CREDENTIAL_MANIFEST_YAML.as_ptr(),
            CREDENTIAL_MANIFEST_YAML.len(),
        )
    };
    assert!(h >= 1, "load_manifest failed: {}", read_err());
    let input = br#"{}"#;
    let ctx = format!(r#"{{"version": 2, "manifest_handle": {}}}"#, h);
    assert_eq!(
        invoke("resolve_credential", Some(input), Some(ctx.as_bytes())),
        0,
        "err={}",
        read_err()
    );
    let out_bytes = read_out();
    let out: serde_json::Value = serde_json::from_slice(&out_bytes).unwrap();
    assert_eq!(out["status"], "missing");
    assert_eq!(out["source_kind"], "none");
    assert_eq!(out["implicit_env_allowed"], false);
    assert_eq!(out["implicit_keyring_allowed"], false);
    let public = String::from_utf8(out_bytes).unwrap();
    assert!(!public.contains("host-env-token-ignored-by-wasm"));
    std::env::remove_var("REPLICATE_API_TOKEN");
}

// ---- WASM-002: memory management ------------------------------------

#[test]
fn test_ailib_free_null_safe() {
    // Must be a no-op and not UB.
    unsafe { ailib_free(std::ptr::null_mut(), 0) };
    unsafe { ailib_free(std::ptr::null_mut(), 123) };
}

#[test]
fn test_ailib_out_consume_ownership_transfer() {
    let _g = test_lock();
    reset_state();
    set_out(b"{\"ok\":true}".to_vec());
    assert!(ailib_out_len() > 0);
    let mut len: usize = 0;
    let ptr = unsafe { ailib_out_consume(&mut len as *mut usize) };
    assert!(!ptr.is_null());
    assert_eq!(len, 11);
    // Post-consume: LAST_OUT is empty again.
    assert_eq!(ailib_out_len(), 0);
    assert!(ailib_out_ptr().is_null());
    // Caller-owned data is readable.
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    assert_eq!(slice, b"{\"ok\":true}");
    // Release.
    unsafe { ailib_free(ptr, len) };
}

#[test]
fn test_ailib_out_consume_empty_returns_null() {
    let _g = test_lock();
    reset_state();
    let mut len: usize = 99;
    let ptr = unsafe { ailib_out_consume(&mut len as *mut usize) };
    assert!(ptr.is_null());
    assert_eq!(len, 0);
}

#[test]
fn test_arena_reset_clears_both_buffers() {
    let _g = test_lock();
    reset_state();
    set_out(b"payload".to_vec());
    set_err("boom");
    assert!(ailib_out_len() > 0);
    ailib_arena_reset();
    assert_eq!(ailib_out_len(), 0);
    // LAST_ERR is drained too.
    assert_eq!(LAST_ERR.lock().map(|g| g.len()).unwrap_or(999), 0);
}

#[test]
fn test_1000_sequential_calls_memory_bounded() {
    // Stress: sequential build+consume+free cycles must not grow
    // LAST_OUT unboundedly. After each cycle LAST_OUT is empty.
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1);
    let req = serde_json::to_vec(&serde_json::json!({
        "model": "m",
        "messages": [{"role":"user","content":"hi"}],
        "stream": false
    }))
    .unwrap();
    for _ in 0..1000 {
        let rc = unsafe { ailib_build_chat_request(h, req.as_ptr(), req.len()) };
        assert_eq!(rc, 0);
        let mut len: usize = 0;
        let ptr = unsafe { ailib_out_consume(&mut len as *mut usize) };
        assert!(!ptr.is_null());
        unsafe { ailib_free(ptr, len) };
        assert_eq!(ailib_out_len(), 0);
    }
}

// ---- WASM-003: snapshot / restore -----------------------------------

#[test]
fn test_snapshot_empty_state() {
    let _g = test_lock();
    reset_state();
    assert_eq!(ailib_snapshot_state(), 0);
    let snap: WasmStateSnapshot = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(snap.abi_version, AILIB_ABI_VERSION);
    assert_eq!(snap.version, SNAPSHOT_FORMAT_VERSION);
    assert!(snap.manifests.is_empty());
}

#[test]
fn test_snapshot_with_manifests() {
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1);
    assert_eq!(ailib_snapshot_state(), 0);
    let snap: WasmStateSnapshot = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(snap.manifests.len(), 1);
    assert_eq!(snap.manifests[0].id, "test-provider");
    assert!(snap.manifests[0].raw_yaml.contains("api.example.com"));
}

#[test]
fn test_restore_roundtrip_manifests() {
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1);
    assert_eq!(ailib_snapshot_state(), 0);
    let snap_bytes = read_out();

    // Drop all state, then restore from snapshot.
    reset_state();
    assert_eq!(
        unsafe { ailib_restore_state(snap_bytes.as_ptr(), snap_bytes.len()) },
        0,
        "err={}",
        read_err()
    );

    // Handle 1 must still work (IDs may be re-assigned but the entry exists).
    let name = b"streaming";
    let rc = unsafe { ailib_check_capability(1, name.as_ptr(), name.len()) };
    assert_eq!(rc, 1, "err={}", read_err());
}

#[test]
fn test_restore_corrupt_snapshot_no_side_effect() {
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1);

    // Malformed snapshot JSON — restore must fail and leave state intact.
    let corrupt =
        br#"{"version": 1, "abi_version": 2, "manifests": [{"id":"x","raw_yaml":"not yaml"}]}"#;
    assert_eq!(
        unsafe { ailib_restore_state(corrupt.as_ptr(), corrupt.len()) },
        -1
    );
    // Original manifest handle still resolves.
    let name = b"streaming";
    let rc = unsafe { ailib_check_capability(1, name.as_ptr(), name.len()) };
    assert_eq!(rc, 1);
}

#[test]
fn test_restore_rejects_newer_abi_version() {
    let _g = test_lock();
    reset_state();
    let future = format!(
        r#"{{"version":1,"abi_version":{},"manifests":[]}}"#,
        AILIB_ABI_VERSION + 1
    );
    assert_eq!(
        unsafe { ailib_restore_state(future.as_ptr(), future.len()) },
        -1
    );
    assert!(read_err().contains("abi_version"));
}

#[test]
fn test_invoke_snapshot_and_restore_ops() {
    let _g = test_lock();
    reset_state();
    let h = unsafe { ailib_load_manifest(MIN_MANIFEST_YAML.as_ptr(), MIN_MANIFEST_YAML.len()) };
    assert!(h >= 1);
    // snapshot via invoke
    assert_eq!(invoke("snapshot_state", None, None), 0);
    let snap = read_out();
    reset_state();
    // restore via invoke
    assert_eq!(invoke("restore_state", Some(&snap), None), 0);
    let v: serde_json::Value = serde_json::from_slice(&read_out()).unwrap();
    assert_eq!(v["restored_manifests"], 1);
}
