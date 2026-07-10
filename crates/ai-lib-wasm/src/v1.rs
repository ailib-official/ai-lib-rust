//! v1 positional WASM C exports (PT-072 Phase 1).
//!
//! 中文：v1 位置参数式 C 导出。

use ai_lib_core::drivers::{create_driver, DriverResponse};
use ai_lib_core::error_code::StandardErrorCode;
use ai_lib_core::protocol::{load_manifest_validated, UnifiedRequest};
use serde::Serialize;

use crate::buffers::*;

// =============================================================================
// V1 positional exports (existing — unchanged signatures; additive-only rule)
// =============================================================================

/// Returns manifest handle (1-based) or 0 on failure. Read `ailib_out_*` / `ailib_err_*`.
#[no_mangle]
pub unsafe extern "C" fn ailib_load_manifest(ptr: *const u8, len: usize) -> u32 {
    clear_err();
    bump_calls();
    let bytes = match bytes_from_ptr(ptr, len) {
        Ok(b) => b.to_vec(),
        Err(e) => {
            set_err(e);
            bump_errors();
            return 0;
        }
    };
    match load_manifest_validated(&bytes) {
        Ok(manifest) => {
            let mut g = match MANIFESTS.lock() {
                Ok(g) => g,
                Err(e) => {
                    set_err(e.to_string());
                    bump_errors();
                    return 0;
                }
            };
            g.push(Some((manifest, bytes)));
            g.len() as u32
        }
        Err(e) => {
            set_err(e.to_string());
            bump_errors();
            0
        }
    }
}

/// 1 if supported, 0 if not, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn ailib_check_capability(
    handle: u32,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    clear_err();
    let name = match str_from_ptr(name_ptr, name_len) {
        Ok(s) => s,
        Err(e) => {
            set_err(e);
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
    let slot = match handle
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
    if slot.supports_capability(name.trim()) {
        1
    } else {
        0
    }
}

/// Build provider chat body JSON from manifest handle + UnifiedRequest-shaped JSON. Output in `ailib_out_*`.
#[no_mangle]
pub unsafe extern "C" fn ailib_build_chat_request(
    handle: u32,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    clear_err();
    bump_calls();
    let json_slice = match bytes_from_ptr(json_ptr, json_len) {
        Ok(b) => b,
        Err(e) => {
            set_err(e);
            bump_errors();
            return -1;
        }
    };
    let req: UnifiedRequest = match serde_json::from_slice::<WasmChatRequest>(json_slice) {
        Ok(r) => r.into(),
        Err(e) => {
            set_err(format!("messages json: {}", e));
            bump_errors();
            return -1;
        }
    };
    let g = match MANIFESTS.lock() {
        Ok(g) => g,
        Err(e) => {
            set_err(e.to_string());
            bump_errors();
            return -1;
        }
    };
    let (m, raw) = match handle
        .checked_sub(1)
        .and_then(|i| g.get(i as usize))
        .and_then(|x| x.as_ref())
    {
        Some(x) => x,
        None => {
            set_err("invalid manifest handle");
            bump_errors();
            return -1;
        }
    };
    let driver = create_driver(
        api_style_from_raw(raw),
        m.id.as_str(),
        caps_from_manifest(m),
    );
    let built = match driver.build_request(
        &req.messages,
        &req.model,
        req.temperature,
        req.max_tokens,
        req.stream,
        None,
    ) {
        Ok(r) => r,
        Err(e) => {
            set_err(e.to_string());
            bump_errors();
            return -1;
        }
    };
    match serde_json::to_vec(&built.body) {
        Ok(v) => {
            set_out(v);
            0
        }
        Err(e) => {
            set_err(e.to_string());
            bump_errors();
            -1
        }
    }
}

#[derive(Serialize)]
struct NormalizedResponse {
    content: Option<String>,
    finish_reason: Option<String>,
    usage: Option<serde_json::Value>,
    tool_calls: Vec<serde_json::Value>,
}

/// Parse provider response JSON using driver for this manifest. Output in `ailib_out_*`.
#[no_mangle]
pub unsafe extern "C" fn ailib_parse_chat_response(
    handle: u32,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    clear_err();
    bump_calls();
    let json_slice = match bytes_from_ptr(json_ptr, json_len) {
        Ok(b) => b,
        Err(e) => {
            set_err(e);
            bump_errors();
            return -1;
        }
    };
    let body: serde_json::Value = match serde_json::from_slice(json_slice) {
        Ok(v) => v,
        Err(e) => {
            set_err(format!("response json: {}", e));
            bump_errors();
            return -1;
        }
    };
    let driver = match driver_for_handle(handle) {
        Ok(d) => d,
        Err(e) => {
            set_err(e);
            bump_errors();
            return -1;
        }
    };
    let DriverResponse {
        content,
        finish_reason,
        usage,
        tool_calls,
        ..
    } = match driver.parse_response(&body) {
        Ok(r) => r,
        Err(e) => {
            set_err(e.to_string());
            bump_errors();
            return -1;
        }
    };
    if let Some(u) = &usage {
        if let Ok(mut m) = METRICS.lock() {
            m.total_tokens_in = m.total_tokens_in.saturating_add(u.prompt_tokens as u64);
            m.total_tokens_out = m
                .total_tokens_out
                .saturating_add(u.completion_tokens as u64);
        }
    }
    let usage_v = usage.map(|u| serde_json::to_value(u).unwrap_or(serde_json::Value::Null));
    let norm = NormalizedResponse {
        content,
        finish_reason,
        usage: usage_v,
        tool_calls,
    };
    match serde_json::to_vec(&norm) {
        Ok(v) => {
            set_out(v);
            0
        }
        Err(e) => {
            set_err(e.to_string());
            bump_errors();
            -1
        }
    }
}

/// Classify HTTP error; writes `{"code":"E...."}` to `ailib_out_*`.
#[no_mangle]
pub unsafe extern "C" fn ailib_classify_error(
    status_code: u16,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    clear_err();
    let code = if json_len == 0 || json_ptr.is_null() {
        StandardErrorCode::from_http_status(status_code)
    } else {
        match bytes_from_ptr(json_ptr, json_len) {
            Ok(b) => match serde_json::from_slice::<serde_json::Value>(b) {
                Ok(v) => {
                    let class = v
                        .pointer("/error/type")
                        .or_else(|| v.get("type"))
                        .and_then(|x: &serde_json::Value| x.as_str())
                        .unwrap_or("");
                    let c = StandardErrorCode::from_error_class(class);
                    if c == StandardErrorCode::Unknown {
                        StandardErrorCode::from_http_status(status_code)
                    } else {
                        c
                    }
                }
                Err(_) => StandardErrorCode::from_http_status(status_code),
            },
            Err(_) => StandardErrorCode::from_http_status(status_code),
        }
    };
    let out = serde_json::json!({ "code": code.code() });
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

/// Extract usage object from response JSON. Output in `ailib_out_*` (may be `{}`).
#[no_mangle]
pub unsafe extern "C" fn ailib_extract_usage(json_ptr: *const u8, json_len: usize) -> i32 {
    clear_err();
    let json_slice = match bytes_from_ptr(json_ptr, json_len) {
        Ok(b) => b,
        Err(e) => {
            set_err(e);
            return -1;
        }
    };
    let body: serde_json::Value = match serde_json::from_slice(json_slice) {
        Ok(v) => v,
        Err(e) => {
            set_err(format!("response json: {}", e));
            return -1;
        }
    };
    let usage = body.get("usage").cloned().unwrap_or(serde_json::json!({}));
    match serde_json::to_vec(&usage) {
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
