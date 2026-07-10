//! ABI version and capability negotiation (WASM-001).
//!
//! 中文：ABI 版本与能力协商。

use std::sync::OnceLock;

use crate::state::SNAPSHOT_FORMAT_VERSION;
use crate::AILIB_ABI_VERSION;

/// Returns the ABI version implemented by this module.
///
/// Hosts should call this first to decide whether to use positional v1 exports
/// or the unified ilib_invoke dispatcher. Unknown or newer versions should
/// fall back to v1 functions (which are guaranteed to exist on all releases).
#[no_mangle]
pub extern "C" fn ailib_abi_version() -> u32 {
    AILIB_ABI_VERSION
}

/// Statically computed capabilities JSON describing which ops ilib_invoke
/// accepts. Cached to avoid per-call allocation.
pub(crate) fn capabilities_json() -> &'static [u8] {
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

/// Pointer to the capabilities JSON (read ilib_capabilities_len bytes).
#[no_mangle]
pub extern "C" fn ailib_capabilities_ptr() -> *const u8 {
    capabilities_json().as_ptr()
}

/// Length of the capabilities JSON in bytes.
#[no_mangle]
pub extern "C" fn ailib_capabilities_len() -> usize {
    capabilities_json().len()
}
