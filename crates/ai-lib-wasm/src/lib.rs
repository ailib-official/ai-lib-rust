//! WASI `wasm32-wasip1` exports for protocol + driver helpers.
//!
//! 中文：薄封装层，供 wasmtime / gateway 等加载；不携带 HTTP 客户端与策略（P 层）依赖。
//!
//! ## ABI Surface (v2)
//!
//! Three generations of exports coexist (all additive, zero breaking change):
//!
//! 1. **v1 positional exports** — `ailib_load_manifest`, `ailib_build_chat_request`,
//!    `ailib_parse_chat_response`, `ailib_classify_error`, `ailib_check_capability`,
//!    `ailib_extract_usage`, plus `ailib_out_*` / `ailib_err_*` accessors.
//!    These remain the primary PT-072 Phase 1 interface.
//!
//! 2. **v2 capability negotiation** (WASM-001) — `ailib_abi_version`,
//!    `ailib_capabilities_ptr` / `ailib_capabilities_len`. Host queries WASM
//!    on load to decide which path to use.
//!
//! 3. **v2 unified dispatcher** (WASM-001) — `ailib_invoke(op, input, ctx)`.
//!    Single entry point with structured JSON args; new ops/fields are
//!    additive. Old callers may stay on positional exports forever.
//!
//! Additional v2 facilities:
//! - **Memory hygiene** (WASM-002) — `ailib_free`, `ailib_out_consume`,
//!   `ailib_arena_reset` (bulk release of `LAST_OUT` / `LAST_ERR`).
//! - **State migration** (WASM-003) — `ailib_snapshot_state`,
//!   `ailib_restore_state` for hot upgrades.

use std::sync::{Mutex, OnceLock};

// credentials helpers inlined below (credentials module is #[cfg(not(target_arch = "wasm32"))])
use ai_lib_core::drivers::{create_driver, DriverResponse, ProviderDriver};
use ai_lib_core::error_code::StandardErrorCode;
use ai_lib_core::protocol::v2::capabilities::{CapabilitiesV2, Capability, LegacyCapabilities};
use ai_lib_core::protocol::v2::manifest::{ApiStyle, ManifestV2};
use ai_lib_core::protocol::{load_manifest_validated, ProtocolManifest, UnifiedRequest};
use ai_lib_core::types::message::Message;
use ai_lib_core::types::tool::ToolDefinition;
use serde::{Deserialize, Serialize};

mod state;
pub use state::{
    ManifestEntry, StreamState, WasmMetrics, WasmStateSnapshot, SNAPSHOT_FORMAT_VERSION,
};

/// Current ABI version reported by `ailib_abi_version()`.
///
/// Bump only on **breaking** semantic changes. New ops on `ailib_invoke` and
/// new optional fields in input / ctx JSON are additive and do NOT require
/// a bump.
pub const AILIB_ABI_VERSION: u32 = 2;

/// JSON shape accepted by `ailib_build_chat_request` (omits `response_format` / full `UnifiedRequest` serde).
#[derive(Debug, Deserialize)]
struct WasmChatRequest {
    #[serde(default)]
    operation: String,
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

impl From<WasmChatRequest> for UnifiedRequest {
    fn from(w: WasmChatRequest) -> Self {
        UnifiedRequest {
            operation: w.operation,
            model: w.model,
            messages: w.messages,
            temperature: w.temperature,
            max_tokens: w.max_tokens,
            stream: w.stream,
            tools: w.tools,
            tool_choice: w.tool_choice,
            response_format: None,
        }
    }
}

static MANIFESTS: Mutex<Vec<Option<(ProtocolManifest, Vec<u8>)>>> = Mutex::new(Vec::new());
static LAST_OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static LAST_ERR: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static METRICS: Mutex<WasmMetrics> = Mutex::new(WasmMetrics {
    total_calls: 0,
    total_errors: 0,
    total_tokens_in: 0,
    total_tokens_out: 0,
});

fn set_out(bytes: Vec<u8>) {
    *LAST_OUT.lock().expect("out lock") = bytes;
}

fn set_err(s: impl AsRef<str>) {
    *LAST_ERR.lock().expect("err lock") = s.as_ref().as_bytes().to_vec();
}

fn clear_err() {
    LAST_ERR.lock().expect("err lock").clear();
}

fn bump_calls() {
    if let Ok(mut m) = METRICS.lock() {
        m.total_calls = m.total_calls.saturating_add(1);
    }
}

fn bump_errors() {
    if let Ok(mut m) = METRICS.lock() {
        m.total_errors = m.total_errors.saturating_add(1);
    }
}

fn caps_from_manifest(m: &ProtocolManifest) -> Vec<Capability> {
    CapabilitiesV2::Legacy(LegacyCapabilities {
        streaming: m.capabilities.streaming,
        tools: m.capabilities.tools,
        vision: m.capabilities.vision,
        agentic: m.capabilities.agentic,
        reasoning: m.capabilities.reasoning,
        parallel_tools: m.capabilities.parallel_tools,
    })
    .all_capabilities()
}

fn api_style_from_raw(bytes: &[u8]) -> ApiStyle {
    serde_yaml::from_slice::<ManifestV2>(bytes)
        .map(|m| m.detect_api_style())
        .unwrap_or(ApiStyle::OpenAiCompatible)
}

fn driver_for_handle(handle: u32) -> Result<Box<dyn ProviderDriver>, String> {
    let g = MANIFESTS.lock().map_err(|e| e.to_string())?;
    let slot = handle
        .checked_sub(1)
        .and_then(|i| g.get(i as usize))
        .ok_or_else(|| "invalid manifest handle".to_string())?;
    let (m, raw) = slot
        .as_ref()
        .ok_or_else(|| "invalid manifest handle".to_string())?;
    let caps = caps_from_manifest(m);
    let style = api_style_from_raw(raw);
    Ok(create_driver(style, m.id.as_str(), caps))
}

unsafe fn bytes_from_ptr<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], String> {
    if ptr.is_null() || len == 0 {
        return Err("null or empty input".to_string());
    }
    Ok(std::slice::from_raw_parts(ptr, len))
}

unsafe fn str_from_ptr(ptr: *const u8, len: usize) -> Result<String, String> {
    let b = bytes_from_ptr(ptr, len)?;
    std::str::from_utf8(b)
        .map(|s| s.to_string())
        .map_err(|e| e.to_string())
}

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

// =============================================================================
// Output / error accessors (v1 — unchanged)
// =============================================================================

#[no_mangle]
pub extern "C" fn ailib_out_ptr() -> *const u8 {
    LAST_OUT
        .lock()
        .ok()
        .and_then(|g| {
            let g = &*g;
            if g.is_empty() {
                None
            } else {
                Some(g.as_ptr())
            }
        })
        .unwrap_or(std::ptr::null())
}

#[no_mangle]
pub extern "C" fn ailib_out_len() -> usize {
    LAST_OUT.lock().map(|g| g.len()).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn ailib_err_ptr() -> *const u8 {
    LAST_ERR
        .lock()
        .ok()
        .and_then(|g| {
            let g = &*g;
            if g.is_empty() {
                None
            } else {
                Some(g.as_ptr())
            }
        })
        .unwrap_or(std::ptr::null())
}

#[no_mangle]
pub extern "C" fn ailib_err_len() -> usize {
    LAST_ERR.lock().map(|g| g.len()).unwrap_or(0)
}

// =============================================================================
// WASM-001: ABI version + capabilities + unified invoke dispatcher
// =============================================================================

/// Returns the ABI version implemented by this module.
///
/// Hosts should call this first to decide whether to use positional v1 exports
/// or the unified `ailib_invoke` dispatcher. Unknown or newer versions should
/// fall back to v1 functions (which are guaranteed to exist on all releases).
#[no_mangle]
pub extern "C" fn ailib_abi_version() -> u32 {
    AILIB_ABI_VERSION
}

/// Statically computed capabilities JSON describing which ops `ailib_invoke`
/// accepts. Cached to avoid per-call allocation.
fn capabilities_json() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let v = serde_json::json!({
                "version": AILIB_ABI_VERSION,
                "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
                "ops": [
                    "abi_version",
                    "capabilities",
                    "load_manifest",
                    "check_capability",
                    "build_request",
                    "parse_response",
                    "classify_error",
                    "extract_usage",
                    "resolve_credential",
                    "snapshot_state",
                    "restore_state",
                    "metrics"
                ],
                "memory": {
                    "ownership_transfer": true,
                    "free": "ailib_free",
                    "consume": "ailib_out_consume",
                    "arena_reset": "ailib_arena_reset"
                },
                "features": {
                    "structured_input": true,
                    "additive_ctx": true,
                    "state_migration": true,
                    "host_supplied_credentials": true
                }
            });
            serde_json::to_vec(&v).expect("serialize static capabilities")
        })
        .as_slice()
}

/// Pointer to the capabilities JSON (read `ailib_capabilities_len` bytes).
#[no_mangle]
pub extern "C" fn ailib_capabilities_ptr() -> *const u8 {
    capabilities_json().as_ptr()
}

/// Length of the capabilities JSON in bytes.
#[no_mangle]
pub extern "C" fn ailib_capabilities_len() -> usize {
    capabilities_json().len()
}

/// Invocation context — additive JSON object.
///
/// Callers at v1 may omit the `version` field (default 1 is accepted).
/// Future fields must be added with `#[serde(default)]` to preserve
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

fn write_out_json(v: &serde_json::Value) -> i32 {
    match serde_json::to_vec(v) {
        Ok(b) => {
            set_out(b);
            0
        }
        Err(e) => {
            set_err(e.to_string());
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
    let Some(auth) = manifest.endpoint.as_ref().and_then(|e| e.auth.as_ref()) else {
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

fn parse_input_json<T: for<'de> Deserialize<'de>>(ptr: *const u8, len: usize) -> Result<T, String> {
    if len == 0 || ptr.is_null() {
        return Err("input required".to_string());
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    serde_json::from_slice::<T>(bytes).map_err(|e| format!("input json: {}", e))
}

// =============================================================================
// WASM-002: Explicit memory management
// =============================================================================

/// Releases a buffer previously handed out via `ailib_out_consume`.
///
/// Safety: `ptr` must point to a `Box<[u8]>` allocation of exactly `len` bytes
/// produced by `ailib_out_consume` (or another WASM export that documents this
/// contract). Passing a null `ptr` is a no-op. Passing a dangling or mismatched
/// allocation is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn ailib_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    // Must match the fat pointer produced by `ailib_out_consume` via
    // `Box::into_raw(boxed) as *mut u8` where `boxed` is `Box<[u8]>`.
    let raw = std::ptr::slice_from_raw_parts_mut(ptr, len);
    drop(Box::from_raw(raw));
}

/// Takes ownership of the current `LAST_OUT` buffer and returns a raw pointer.
///
/// Writes the length into `*out_len` (if non-null) and clears `LAST_OUT` so
/// subsequent `ailib_out_ptr` / `ailib_out_len` return null/0. The caller owns
/// the returned buffer and MUST release it via `ailib_free(ptr, len)`.
///
/// Returns `null` when `LAST_OUT` is empty (nothing to consume).
#[no_mangle]
pub unsafe extern "C" fn ailib_out_consume(out_len: *mut usize) -> *mut u8 {
    let taken: Vec<u8> = match LAST_OUT.lock() {
        Ok(mut g) => std::mem::take(&mut *g),
        Err(_) => {
            if !out_len.is_null() {
                *out_len = 0;
            }
            return std::ptr::null_mut();
        }
    };
    if taken.is_empty() {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return std::ptr::null_mut();
    }
    let boxed: Box<[u8]> = taken.into_boxed_slice();
    let len = boxed.len();
    if !out_len.is_null() {
        *out_len = len;
    }
    Box::into_raw(boxed) as *mut u8
}

/// Bulk-release: drops `LAST_OUT` and `LAST_ERR` buffers.
///
/// Equivalent to `reset()` on the task-spec's arena: after this call the
/// resident scratch buffers are empty (their `Vec` backings dropped, so the
/// allocator may reuse the pages). A true pre-allocated arena is deferred as
/// a future optimization — the explicit-free pattern meets the bounded-memory
/// acceptance criteria for the current workload (1-4KB per call).
#[no_mangle]
pub extern "C" fn ailib_arena_reset() {
    if let Ok(mut g) = LAST_OUT.lock() {
        *g = Vec::new();
    }
    if let Ok(mut g) = LAST_ERR.lock() {
        *g = Vec::new();
    }
}

// =============================================================================
// WASM-003: Atomic state migration (snapshot / restore)
// =============================================================================

fn snapshot_state_build() -> WasmStateSnapshot {
    let metrics = METRICS.lock().map(|g| g.clone()).unwrap_or_default();
    let manifests: Vec<ManifestEntry> = MANIFESTS
        .lock()
        .map(|g| {
            g.iter()
                .filter_map(|s| s.as_ref())
                .map(|(m, raw)| ManifestEntry {
                    id: m.id.clone(),
                    raw_yaml: String::from_utf8_lossy(raw).into_owned(),
                    loaded_at: 0,
                })
                .collect()
        })
        .unwrap_or_default();
    WasmStateSnapshot {
        version: SNAPSHOT_FORMAT_VERSION,
        abi_version: AILIB_ABI_VERSION,
        manifests,
        active_streams: Vec::new(),
        metrics,
    }
}

fn ailib_snapshot_state_inner() -> i32 {
    let snap = snapshot_state_build();
    match serde_json::to_vec(&snap) {
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

unsafe fn ailib_restore_state_inner(ptr: *const u8, len: usize) -> i32 {
    let bytes = match bytes_from_ptr(ptr, len) {
        Ok(b) => b,
        Err(e) => {
            set_err(e);
            return -1;
        }
    };
    let snap: WasmStateSnapshot = match serde_json::from_slice(bytes) {
        Ok(s) => s,
        Err(e) => {
            set_err(format!("snapshot parse: {}", e));
            return -1;
        }
    };
    if snap.abi_version > AILIB_ABI_VERSION {
        set_err(format!(
            "snapshot abi_version {} newer than this WASM ({})",
            snap.abi_version, AILIB_ABI_VERSION
        ));
        return -1;
    }
    if snap.version > SNAPSHOT_FORMAT_VERSION {
        set_err(format!(
            "snapshot format version {} newer than supported ({})",
            snap.version, SNAPSHOT_FORMAT_VERSION
        ));
        return -1;
    }
    // Atomic restore: pre-validate ALL manifests before swapping. If any fails,
    // return -1 and leave MANIFESTS untouched.
    let mut new_manifests: Vec<Option<(ProtocolManifest, Vec<u8>)>> =
        Vec::with_capacity(snap.manifests.len());
    for entry in &snap.manifests {
        let raw = entry.raw_yaml.as_bytes().to_vec();
        match load_manifest_validated(&raw) {
            Ok(m) => new_manifests.push(Some((m, raw))),
            Err(e) => {
                set_err(format!("restore manifest '{}': {}", entry.id, e));
                return -1;
            }
        }
    }
    // Swap all state under the manifests lock so partial visibility is impossible.
    let mut g = match MANIFESTS.lock() {
        Ok(g) => g,
        Err(e) => {
            set_err(e.to_string());
            return -1;
        }
    };
    *g = new_manifests;
    if let Ok(mut m) = METRICS.lock() {
        *m = snap.metrics.clone();
    }
    write_out_json(&serde_json::json!({
        "restored_manifests": snap.manifests.len(),
        "needs_replay_streams": snap.active_streams.iter().filter(|s| s.needs_replay).count(),
    }))
}

/// Serialize all internal state to JSON. Output in `ailib_out_*` (consume via
/// `ailib_out_consume` to transfer ownership).
#[no_mangle]
pub extern "C" fn ailib_snapshot_state() -> i32 {
    clear_err();
    ailib_snapshot_state_inner()
}

/// Atomically restore a snapshot previously produced by `ailib_snapshot_state`.
///
/// Returns `0` on success, `-1` on error. On error, the module's state is
/// guaranteed to be untouched.
#[no_mangle]
pub unsafe extern "C" fn ailib_restore_state(ptr: *const u8, len: usize) -> i32 {
    clear_err();
    ailib_restore_state_inner(ptr, len)
}

// =============================================================================
// Tests (native target)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
