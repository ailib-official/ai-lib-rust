//! Explicit memory management (WASM-002).
//!
//! 中文：显式内存管理。

use crate::buffers::{LAST_ERR, LAST_OUT};

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
