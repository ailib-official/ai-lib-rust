//! WASM internal state snapshot & restore types (WASM-003).
//!
//! 中文：WASM 模块内部状态的可序列化快照。用于热升级（hot upgrade）场景：
//! - 宿主侧 (wasmtime / gateway) 调用 `ailib_snapshot_state` 取出旧实例的状态；
//! - 加载新的 WASM 实例；
//! - 调用 `ailib_restore_state` 注入快照。
//!
//! 当前 ai-lib-wasm 执行层基本是无状态纯函数，唯一持久化的状态是通过
//! `ailib_load_manifest` 注册的 provider manifest 列表，以及累计的调用指标。
//! 活跃的 SSE 流会话由宿主持有，不在本模块；因此 `active_streams` 字段预留
//! 为空（`needs_replay` 策略由宿主执行）。
//!
//! ## Schema versioning
//!
//! `snapshot_format_version` 独立于 `abi_version`：
//! - 新版本理解旧版本（向后兼容）；
//! - 旧版本对陌生字段使用 `serde(default)` 容忍；
//! - 新版本字段缺失时使用默认值。
//!
//! 该原则参考 WASM-003 任务文档中的"新理解旧，旧拒绝新"策略。

use serde::{Deserialize, Serialize};

/// Current snapshot format version.
///
/// Bump when adding non-optional fields; additive optional fields don't
/// require a bump because readers use `#[serde(default)]`.
pub const SNAPSHOT_FORMAT_VERSION: u32 = 1;

/// A single loaded manifest's snapshot entry. We store the raw YAML bytes
/// (as UTF-8 text) rather than the parsed `ProtocolManifest` so that the
/// receiving instance can re-run `load_manifest_validated`, which ensures
/// schema compatibility even across ABI versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub id: String,
    /// Original YAML source (UTF-8 text).
    pub raw_yaml: String,
    /// Unix seconds when the manifest was first loaded; 0 if unknown.
    #[serde(default)]
    pub loaded_at: u64,
}

/// Accumulated WASM-side metrics — survives hot upgrades so the gateway can
/// produce continuous telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WasmMetrics {
    #[serde(default)]
    pub total_calls: u64,
    #[serde(default)]
    pub total_errors: u64,
    #[serde(default)]
    pub total_tokens_in: u64,
    #[serde(default)]
    pub total_tokens_out: u64,
}

/// Placeholder for in-flight stream sessions. The current execution layer is
/// stateless (all stream accumulation lives in the host), so this field is
/// always empty in v1. Reserved to keep the snapshot schema stable when
/// stream state migrates into WASM in a future phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamState {
    pub stream_id: String,
    pub provider_id: String,
    pub model: String,
    #[serde(default)]
    pub accumulated_content: String,
    #[serde(default)]
    pub accumulated_thinking: String,
    #[serde(default)]
    pub tokens_so_far: u64,
    #[serde(default)]
    pub last_event_index: u32,
    /// `true` if the host must re-request the upstream provider on resume;
    /// always true today because the HTTP connection died with the old instance.
    #[serde(default)]
    pub needs_replay: bool,
    #[serde(default)]
    pub started_at: u64,
}

/// Full snapshot of WASM-internal state. Atomic restore contract: if any
/// manifest fails to re-validate, the restore aborts and the original state
/// is untouched (see `ailib_restore_state`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WasmStateSnapshot {
    /// Snapshot schema version (see `SNAPSHOT_FORMAT_VERSION`).
    pub version: u32,
    /// ABI version of the WASM that produced this snapshot.
    pub abi_version: u32,
    #[serde(default)]
    pub manifests: Vec<ManifestEntry>,
    #[serde(default)]
    pub active_streams: Vec<StreamState>,
    #[serde(default)]
    pub metrics: WasmMetrics,
}

impl WasmStateSnapshot {
    pub fn new(abi_version: u32) -> Self {
        Self {
            version: SNAPSHOT_FORMAT_VERSION,
            abi_version,
            manifests: Vec::new(),
            active_streams: Vec::new(),
            metrics: WasmMetrics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_empty() {
        let snap = WasmStateSnapshot::new(2);
        let bytes = serde_json::to_vec(&snap).unwrap();
        let back: WasmStateSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(snap, back);
    }

    #[test]
    fn snapshot_tolerates_unknown_fields() {
        // Forward-compat: snapshots produced by a newer WASM with extra
        // fields must still deserialize here.
        let raw = r#"{
            "version": 1,
            "abi_version": 2,
            "manifests": [],
            "future_field": "ignored",
            "metrics": {"total_calls": 5, "new_counter": 99}
        }"#;
        let snap: WasmStateSnapshot = serde_json::from_str(raw).unwrap();
        assert_eq!(snap.metrics.total_calls, 5);
    }

    #[test]
    fn snapshot_missing_optional_fields_uses_defaults() {
        // Back-compat: a v1 writer may omit optional fields.
        let raw = r#"{"version":1,"abi_version":2}"#;
        let snap: WasmStateSnapshot = serde_json::from_str(raw).unwrap();
        assert!(snap.manifests.is_empty());
        assert!(snap.active_streams.is_empty());
        assert_eq!(snap.metrics.total_calls, 0);
    }
}
