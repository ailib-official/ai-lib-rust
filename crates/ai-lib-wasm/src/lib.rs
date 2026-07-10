//! WASI wasm32-wasip1 exports for protocol + driver helpers.
//!
//! 中文：薄封装层，供 wasmtime / gateway 等加载；不携带 HTTP 客户端与策略（P 层）依赖。
//!
//! ## ABI Surface (v2)
//!
//! Three generations of exports coexist (all additive, zero breaking change):
//!
//! 1. **v1 positional exports** — ilib_load_manifest, ilib_build_chat_request,
//!    ilib_parse_chat_response, ilib_classify_error, ilib_check_capability,
//!    ilib_extract_usage, plus ilib_out_* / ilib_err_* accessors.
//!    These remain the primary PT-072 Phase 1 interface.
//!
//! 2. **v2 capability negotiation** (WASM-001) — ilib_abi_version,
//!    ilib_capabilities_ptr / ilib_capabilities_len. Host queries WASM
//!    on load to decide which path to use.
//!
//! 3. **v2 unified dispatcher** (WASM-001) — ilib_invoke(op, input, ctx).
//!    Single entry point with structured JSON args; new ops/fields are
//!    additive. Old callers may stay on positional exports forever.
//!
//! Additional v2 facilities:
//! - **Memory hygiene** (WASM-002) — ilib_free, ilib_out_consume,
//!   ilib_arena_reset (bulk release of LAST_OUT / LAST_ERR).
//! - **State migration** (WASM-003) — ilib_snapshot_state,
//!   ilib_restore_state for hot upgrades.

mod abi;
mod buffers;
mod invoke;
mod io;
mod memory;
mod snapshot_ops;
mod state;
mod v1;

#[cfg(test)]
mod native_tests;

pub use state::{
    ManifestEntry, StreamState, WasmMetrics, WasmStateSnapshot, SNAPSHOT_FORMAT_VERSION,
};

/// Current ABI version reported by ilib_abi_version().
///
/// Bump only on **breaking** semantic changes. New ops on ilib_invoke and
/// new optional fields in input / ctx JSON are additive and do NOT require
/// a bump.
pub const AILIB_ABI_VERSION: u32 = 2;
