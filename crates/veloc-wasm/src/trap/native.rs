use super::MemoryRange;
#[cfg(veloc_native_traps)]
use crate::vm::TrapCode;
use crate::{
    error::{Error, Result},
    vm::VMContext,
};

pub(crate) const SUPPORTED: bool = cfg!(veloc_native_traps);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CodeRange {
    pub start: usize,
    pub end: usize,
}

#[cfg(veloc_native_traps)]
unsafe extern "C" {
    fn veloc_native_install() -> i32;
    fn veloc_native_call(
        entry: *const u8,
        vmctx: *mut core::ffi::c_void,
        args: *const i64,
        results: *mut i64,
        return_bits: *mut i64,
        initialize: i32,
        memories: *const MemoryRange,
        memory_count: usize,
        code: *const CodeRange,
        code_count: usize,
        memory_trap: i32,
    ) -> i32;
    fn veloc_native_raise(code: i32);
    fn veloc_native_suspend() -> *mut core::ffi::c_void;
    fn veloc_native_restore(scope: *mut core::ffi::c_void);
    #[cfg(all(target_arch = "x86_64", target_env = "gnu"))]
    pub(super) fn veloc_native_handle_signal(
        signal: i32,
        info: *mut libc::siginfo_t,
        context: *mut core::ffi::c_void,
    );
}

pub(crate) fn install() -> Result<()> {
    #[cfg(veloc_native_traps)]
    {
        static INSTALLED: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        let errno = *INSTALLED.get_or_init(|| unsafe { veloc_native_install() });
        if errno == 0 {
            Ok(())
        } else {
            Err(Error::Memory(format!(
                "install native memory traps: {}",
                std::io::Error::from_raw_os_error(errno)
            )))
        }
    }
    #[cfg(not(veloc_native_traps))]
    {
        Err(Error::Unsupported(
            "native memory traps are unavailable on this platform".into(),
        ))
    }
}

/// Caller keeps the loaded code and all memory mappings alive until this returns.
pub(crate) unsafe fn call(
    entry: *const u8,
    vmctx: *mut VMContext,
    args: &[i64],
    results: &mut [i64],
    initialize: bool,
    memories: &[MemoryRange],
    code: &[CodeRange],
) -> core::result::Result<i64, u32> {
    #[cfg(veloc_native_traps)]
    {
        let mut bits = 0;
        let trap = unsafe {
            veloc_native_call(
                entry,
                vmctx.cast(),
                args.as_ptr(),
                results.as_mut_ptr(),
                &mut bits,
                i32::from(initialize),
                memories.as_ptr(),
                memories.len(),
                code.as_ptr(),
                code.len(),
                TrapCode::MemoryOutOfBounds as i32 + 1,
            )
        };
        if trap == 0 {
            Ok(bits)
        } else {
            Err(trap as u32)
        }
    }
    #[cfg(not(veloc_native_traps))]
    {
        let _ = (memories, code);
        if initialize {
            let f: extern "C" fn(*mut VMContext) = unsafe { core::mem::transmute(entry) };
            f(vmctx);
            Ok(0)
        } else {
            let f: extern "C" fn(*mut VMContext, *const i64, *mut i64) -> i64 =
                unsafe { core::mem::transmute(entry) };
            Ok(f(vmctx, args.as_ptr(), results.as_mut_ptr()))
        }
    }
}

/// Returns only when execution is not inside a native call boundary.
pub(crate) unsafe fn raise(code: u32) {
    #[cfg(veloc_native_traps)]
    unsafe {
        veloc_native_raise(code as i32);
    }
    #[cfg(not(veloc_native_traps))]
    let _ = code;
}

/// Interpreter calls must not accidentally jump across their Rust frames to an
/// enclosing native invocation (for example during a reentrant host callback).
pub(crate) fn suspend<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(veloc_native_traps)]
    {
        struct Guard(*mut core::ffi::c_void);
        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe {
                    veloc_native_restore(self.0);
                }
            }
        }
        let _guard = Guard(unsafe { veloc_native_suspend() });
        f()
    }
    #[cfg(not(veloc_native_traps))]
    f()
}
