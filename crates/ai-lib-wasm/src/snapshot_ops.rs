//! State snapshot / restore ops (WASM-003).
//!
//! 中文：状态快照与恢复。

use ai_lib_core::protocol::{load_manifest_validated, ProtocolManifest};

use crate::buffers::{
    bytes_from_ptr, clear_err, set_err, set_out, write_out_json, MANIFESTS, METRICS,
};
use crate::state::{ManifestEntry, WasmStateSnapshot, SNAPSHOT_FORMAT_VERSION};
use crate::AILIB_ABI_VERSION;

// =============================================================================
// WASM-003: Atomic state migration (snapshot / restore)
// =============================================================================

pub(crate) fn snapshot_state_build() -> WasmStateSnapshot {
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

pub(crate) fn ailib_snapshot_state_inner() -> i32 {
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

pub(crate) unsafe fn ailib_restore_state_inner(ptr: *const u8, len: usize) -> i32 {
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
