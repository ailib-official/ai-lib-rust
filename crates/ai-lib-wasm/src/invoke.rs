//! Unified ailib_invoke dispatcher (WASM-001).
//!
//! 中文：统一 ailib_invoke 分发器。

use ai_lib_core::protocol::ProtocolManifest;
use serde::{Deserialize, Serialize};

use crate::abi::capabilities_json;
use crate::buffers::*;
use crate::snapshot_ops::{ailib_restore_state_inner, ailib_snapshot_state_inner};
use crate::v1::{
    ailib_build_chat_request, ailib_check_capability, ailib_classify_error, ailib_extract_usage,
    ailib_load_manifest, ailib_parse_chat_response,
};
use crate::AILIB_ABI_VERSION;

/// forward/backward compatibility.
#[derive(Debug, Clone, Deserialize)]
struct InvokeCtx {
    #[serde(default = "default_ctx_version")]
    version: u32,
    #[serde(default)]
    manifest_handle: Option<u32>,
    #[serde(default)]
    status_code: Option<u16>,
}

fn default_ctx_version() -> u32 {
    1
}

impl Default for InvokeCtx {
    fn default() -> Self {
        Self {
            version: default_ctx_version(),
            manifest_handle: None,
            status_code: None,
        }
    }
}

/// Unified versioned entry point (WASM-001).
///
/// Contract:
/// - `op_ptr/op_len` — UTF-8 operation name (see `capabilities.ops`).
/// - `input_ptr/input_len` — op-specific JSON (may be null/0).
/// - `ctx_ptr/ctx_len` — invocation context JSON (may be null/0; defaults applied).
/// - Return `0` on success (read `ailib_out_*`), `-1` on error (read `ailib_err_*`).
///
/// The dispatcher rejects `ctx.version > AILIB_ABI_VERSION` — the host is
/// expected to step down to a lower version if the WASM is older.
#[no_mangle]
pub unsafe extern "C" fn ailib_invoke(
    op_ptr: *const u8,
    op_len: usize,
    input_ptr: *const u8,
    input_len: usize,
    ctx_ptr: *const u8,
    ctx_len: usize,
) -> i32 {
    clear_err();
    let op = match str_from_ptr(op_ptr, op_len) {
        Ok(s) => s,
        Err(e) => {
            set_err(format!("op: {}", e));
            return -1;
        }
    };
    let ctx: InvokeCtx = if ctx_len == 0 || ctx_ptr.is_null() {
        InvokeCtx::default()
    } else {
        let raw = match bytes_from_ptr(ctx_ptr, ctx_len) {
            Ok(b) => b,
            Err(e) => {
                set_err(format!("ctx: {}", e));
                return -1;
            }
        };
        match serde_json::from_slice::<InvokeCtx>(raw) {
            Ok(c) => c,
            Err(e) => {
                set_err(format!("ctx json: {}", e));
                return -1;
            }
        }
    };
    if ctx.version > AILIB_ABI_VERSION {
        set_err(format!(
            "ctx version {} newer than AILIB_ABI_VERSION {}",
            ctx.version, AILIB_ABI_VERSION
        ));
        return -1;
    }

    match op.as_str() {
        "abi_version" => {
            let out = serde_json::json!({ "version": AILIB_ABI_VERSION });
            write_out_json(&out)
        }
        "capabilities" => {
            set_out(capabilities_json().to_vec());
            0
        }
        "metrics" => {
            let m = METRICS.lock().map(|g| g.clone()).unwrap_or_default();
            match serde_json::to_value(&m).and_then(|v| serde_json::to_vec(&v)) {
                Ok(v) => {
                    set_out(v);
                    0
                }
                Err(e) => {
                    set_err(e.to_string());
                    -1
                }
            }
        }
        "load_manifest" => ailib_load_manifest_bytes(input_ptr, input_len),
        "check_capability" => ailib_invoke_check_capability(&ctx, input_ptr, input_len),
        "build_request" => ailib_invoke_build_request(&ctx, input_ptr, input_len),
        "parse_response" => ailib_invoke_parse_response(&ctx, input_ptr, input_len),
        "classify_error" => ailib_invoke_classify_error(&ctx, input_ptr, input_len),
        "extract_usage" => ailib_extract_usage(input_ptr, input_len),
        "resolve_credential" => ailib_invoke_resolve_credential(&ctx, input_ptr, input_len),
        "snapshot_state" => ailib_snapshot_state_inner(),
        "restore_state" => ailib_restore_state_inner(input_ptr, input_len),
        other => {
            set_err(format!("unknown op: {}", other));
            -1
        }
    }
}
unsafe fn ailib_load_manifest_bytes(ptr: *const u8, len: usize) -> i32 {
    let h = ailib_load_manifest(ptr, len);
    if h == 0 {
        -1
    } else {
        write_out_json(&serde_json::json!({ "handle": h }))
    }
}

unsafe fn ailib_invoke_check_capability(
    ctx: &InvokeCtx,
    input_ptr: *const u8,
    input_len: usize,
) -> i32 {
    #[derive(Deserialize)]
    struct In {
        #[serde(default)]
        handle: Option<u32>,
        name: String,
    }
    let input: In = match parse_input_json(input_ptr, input_len) {
        Ok(v) => v,
        Err(e) => {
            set_err(e);
            return -1;
        }
    };
    let handle = match input.handle.or(ctx.manifest_handle) {
        Some(h) => h,
        None => {
            set_err("check_capability: missing manifest handle (in ctx or input)");
            return -1;
        }
    };
    let name_bytes = input.name.as_bytes();
    let supported = ailib_check_capability(handle, name_bytes.as_ptr(), name_bytes.len());
    if supported < 0 {
        return -1;
    }
    write_out_json(&serde_json::json!({ "supported": supported == 1 }))
}

unsafe fn ailib_invoke_build_request(
    ctx: &InvokeCtx,
    input_ptr: *const u8,
    input_len: usize,
) -> i32 {
    let handle = match ctx.manifest_handle {
        Some(h) => h,
        None => {
            set_err("build_request: ctx.manifest_handle required");
            return -1;
        }
    };
    ailib_build_chat_request(handle, input_ptr, input_len)
}

unsafe fn ailib_invoke_parse_response(
    ctx: &InvokeCtx,
    input_ptr: *const u8,
    input_len: usize,
) -> i32 {
    let handle = match ctx.manifest_handle {
        Some(h) => h,
        None => {
            set_err("parse_response: ctx.manifest_handle required");
            return -1;
        }
    };
    ailib_parse_chat_response(handle, input_ptr, input_len)
}

unsafe fn ailib_invoke_classify_error(
    ctx: &InvokeCtx,
    input_ptr: *const u8,
    input_len: usize,
) -> i32 {
    let status = match ctx.status_code {
        Some(s) => s,
        None => {
            // Allow status_code in input as a fallback
            let parsed: serde_json::Value = if input_len == 0 || input_ptr.is_null() {
                serde_json::Value::Null
            } else {
                match bytes_from_ptr(input_ptr, input_len)
                    .and_then(|b| serde_json::from_slice(b).map_err(|e| e.to_string()))
                {
                    Ok(v) => v,
                    Err(e) => {
                        set_err(format!("classify_error input: {}", e));
                        return -1;
                    }
                }
            };
            match parsed.get("status_code").and_then(|v| v.as_u64()) {
                Some(s) if s <= u16::MAX as u64 => s as u16,
                _ => {
                    set_err("classify_error: status_code required in ctx or input");
                    return -1;
                }
            }
        }
    };
    ailib_classify_error(status, input_ptr, input_len)
}

fn wasm_required_envs(manifest: &ProtocolManifest) -> Vec<String> {
    let Some(auth) = manifest.endpoint.auth.as_ref().or(manifest.auth.as_ref()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(env) = auth.token_env.as_ref().or(auth.key_env.as_ref()) {
        let trimmed = env.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    out
}

fn wasm_provider_id(manifest: &ProtocolManifest) -> &str {
    manifest.provider_id.as_deref().unwrap_or(&manifest.id)
}

fn wasm_conventional_envs(provider_id: &str) -> Vec<String> {
    let normalized = provider_id.to_uppercase().replace('-', "_");
    vec![format!("{normalized}_API_KEY")]
}

unsafe fn ailib_invoke_resolve_credential(
    ctx: &InvokeCtx,
    input_ptr: *const u8,
    input_len: usize,
) -> i32 {
    #[derive(Deserialize)]
    struct In {
        #[serde(default)]
        handle: Option<u32>,
        #[serde(default)]
        explicit_credential: Option<String>,
    }
    #[derive(Serialize)]
    struct Out {
        status: &'static str,
        source_kind: &'static str,
        source_name: Option<&'static str>,
        required: Vec<String>,
        conventional_fallbacks: Vec<String>,
        implicit_env_allowed: bool,
        implicit_keyring_allowed: bool,
        value_redacted: bool,
    }

    let input: In = match parse_input_json(input_ptr, input_len) {
        Ok(v) => v,
        Err(e) => {
            set_err(e);
            return -1;
        }
    };
    let handle = match input.handle.or(ctx.manifest_handle) {
        Some(h) => h,
        None => {
            set_err("resolve_credential: missing manifest handle (in ctx or input)");
            return -1;
        }
    };
    let g = match MANIFESTS.lock() {
        Ok(g) => g,
        Err(e) => {
            set_err(e.to_string());
            return -1;
        }
    };
    let manifest = match handle
        .checked_sub(1)
        .and_then(|i| g.get(i as usize))
        .and_then(|x| x.as_ref())
    {
        Some((m, _)) => m,
        None => {
            set_err("invalid manifest handle");
            return -1;
        }
    };
    let required = wasm_required_envs(manifest);
    let conventional_fallbacks = wasm_conventional_envs(wasm_provider_id(manifest));
    let has_explicit = input
        .explicit_credential
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    let out = if has_explicit {
        Out {
            status: "available",
            source_kind: "explicit",
            source_name: Some("host_supplied"),
            required,
            conventional_fallbacks,
            implicit_env_allowed: false,
            implicit_keyring_allowed: false,
            value_redacted: true,
        }
    } else {
        Out {
            status: "missing",
            source_kind: "none",
            source_name: None,
            required,
            conventional_fallbacks,
            implicit_env_allowed: false,
            implicit_keyring_allowed: false,
            value_redacted: true,
        }
    };
    match serde_json::to_vec(&out) {
        Ok(v) => {
            set_out(v);
            0
        }
        Err(e) => {
            set_err(e.to_string());
            -1
        }
    }
}
