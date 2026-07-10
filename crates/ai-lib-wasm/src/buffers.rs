//! Shared WASM buffers, metrics, and pointer helpers.
//!
//! 中文：WASM 全局缓冲、指标与指针辅助。

use std::sync::Mutex;

use ai_lib_core::drivers::{create_driver, ProviderDriver};
use ai_lib_core::protocol::v2::capabilities::{CapabilitiesV2, Capability, LegacyCapabilities};
use ai_lib_core::protocol::v2::manifest::{ApiStyle, ManifestV2};
use ai_lib_core::protocol::{ProtocolManifest, UnifiedRequest};
use ai_lib_core::types::message::Message;
use ai_lib_core::types::tool::ToolDefinition;
use serde::Deserialize;

use crate::state::WasmMetrics;
/// JSON shape accepted by `ailib_build_chat_request` (omits `response_format` / full `UnifiedRequest` serde).
#[derive(Debug, Deserialize)]
pub(crate) struct WasmChatRequest {
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

pub(crate) static MANIFESTS: Mutex<Vec<Option<(ProtocolManifest, Vec<u8>)>>> =
    Mutex::new(Vec::new());
pub(crate) static LAST_OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
pub(crate) static LAST_ERR: Mutex<Vec<u8>> = Mutex::new(Vec::new());
pub(crate) static METRICS: Mutex<WasmMetrics> = Mutex::new(WasmMetrics {
    total_calls: 0,
    total_errors: 0,
    total_tokens_in: 0,
    total_tokens_out: 0,
});

pub(crate) fn set_out(bytes: Vec<u8>) {
    *LAST_OUT.lock().unwrap_or_else(|e| e.into_inner()) = bytes;
}

pub(crate) fn set_err(s: impl AsRef<str>) {
    *LAST_ERR.lock().unwrap_or_else(|e| e.into_inner()) = s.as_ref().as_bytes().to_vec();
}

pub(crate) fn clear_err() {
    LAST_ERR.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

pub(crate) fn bump_calls() {
    if let Ok(mut m) = METRICS.lock() {
        m.total_calls = m.total_calls.saturating_add(1);
    }
}

pub(crate) fn bump_errors() {
    if let Ok(mut m) = METRICS.lock() {
        m.total_errors = m.total_errors.saturating_add(1);
    }
}

pub(crate) fn caps_from_manifest(m: &ProtocolManifest) -> Vec<Capability> {
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

pub(crate) fn api_style_from_raw(bytes: &[u8]) -> ApiStyle {
    serde_yaml::from_slice::<ManifestV2>(bytes)
        .map(|m| m.detect_api_style())
        .unwrap_or(ApiStyle::OpenAiCompatible)
}

pub(crate) fn driver_for_handle(handle: u32) -> Result<Box<dyn ProviderDriver>, String> {
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

pub(crate) unsafe fn bytes_from_ptr<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], String> {
    if ptr.is_null() || len == 0 {
        return Err("null or empty input".to_string());
    }
    Ok(std::slice::from_raw_parts(ptr, len))
}

pub(crate) unsafe fn str_from_ptr(ptr: *const u8, len: usize) -> Result<String, String> {
    let b = bytes_from_ptr(ptr, len)?;
    std::str::from_utf8(b)
        .map(|s| s.to_string())
        .map_err(|e| e.to_string())
}

pub(crate) fn write_out_json(v: &serde_json::Value) -> i32 {
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

pub(crate) fn parse_input_json<T: for<'de> Deserialize<'de>>(
    ptr: *const u8,
    len: usize,
) -> Result<T, String> {
    if len == 0 || ptr.is_null() {
        return Err("input required".to_string());
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    serde_json::from_slice::<T>(bytes).map_err(|e| format!("input json: {}", e))
}
