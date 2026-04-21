# Changelog — `ai-lib-wasm`

## 0.9.5

- WASM-001: `ailib_abi_version`, `ailib_capabilities_*`, unified `ailib_invoke` dispatcher; v1 exports retained.
- WASM-002: `ailib_free`, `ailib_out_consume`, `ailib_arena_reset` for explicit buffer lifecycle.
- WASM-003: `ailib_snapshot_state` / `ailib_restore_state` with `WasmStateSnapshot` JSON (`state.rs`).
- Internal unit tests for ABI, memory transfer, and snapshot roundtrips.

## 0.9.4

- Baseline WASI exports for manifest loading, request build, response parse, and error classification (PT-072 Phase 1).
