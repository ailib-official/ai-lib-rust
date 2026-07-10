//! Output / error buffer accessors.
//!
//! 中文：输出与错误缓冲访问器。

use crate::buffers::{LAST_ERR, LAST_OUT};

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
