use crate::error::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) struct MemoryRange {
    pub start: usize,
    pub accessible_end: usize,
    pub reservation_end: usize,
}

impl MemoryRange {
    pub(crate) fn new(memory: &crate::vm::VMMemory) -> Self {
        let (start, reservation_end) = memory.reservation();
        Self {
            start,
            accessible_end: start + memory.current_length,
            reservation_end,
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
mod platform {
    use super::*;
    use core::cell::Cell;
    use core::ffi::c_void;
    use core::ptr;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::OnceLock;

    struct Scope {
        ranges: Vec<MemoryRange>,
        previous: *const Scope,
    }

    thread_local! {
        static ACTIVE: Cell<*const Scope> = const { Cell::new(ptr::null()) };
    }

    struct Previous {
        segv: libc::sigaction,
        bus: libc::sigaction,
    }

    static PREVIOUS: OnceLock<Previous> = OnceLock::new();
    static INSTALL_RESULT: OnceLock<core::result::Result<(), i32>> = OnceLock::new();

    pub(super) fn enabled() -> bool {
        INSTALL_RESULT.get().is_some_and(|result| result.is_ok())
    }

    #[derive(Debug)]
    struct MemoryFault(i32);

    pub(super) fn install() -> Result<()> {
        match INSTALL_RESULT.get_or_init(|| unsafe {
            let mut action: libc::sigaction = core::mem::zeroed();
            action.sa_sigaction = handle_signal as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO;
            libc::sigemptyset(&mut action.sa_mask);
            let mut segv = core::mem::zeroed();
            let mut bus = core::mem::zeroed();
            if libc::sigaction(libc::SIGSEGV, ptr::null(), &mut segv) != 0
                || libc::sigaction(libc::SIGBUS, ptr::null(), &mut bus) != 0
            {
                return Err(*libc::__errno_location());
            }
            // Publish the previous handlers before either new handler is
            // visible to a signal delivered on another thread.
            let _ = PREVIOUS.set(Previous { segv, bus });
            if libc::sigaction(libc::SIGSEGV, &action, ptr::null_mut()) != 0 {
                return Err(*libc::__errno_location());
            }
            if libc::sigaction(libc::SIGBUS, &action, ptr::null_mut()) != 0 {
                let error = *libc::__errno_location();
                libc::sigaction(
                    libc::SIGSEGV,
                    &PREVIOUS.get().unwrap().segv,
                    ptr::null_mut(),
                );
                return Err(error);
            }
            Ok(())
        }) {
            Ok(_) => Ok(()),
            Err(errno) => Err(Error::Memory(format!(
                "Installing memory trap handler failed (errno {errno})"
            ))),
        }
    }

    // The runtime only installs this on glibc/x86-64, where the signal
    // trampoline has unwind information. The caught payload is private to
    // veloc-wasm; ordinary host panics are always resumed unchanged.
    unsafe extern "C-unwind" fn handle_signal(
        signal: i32,
        info: *mut libc::siginfo_t,
        context: *mut c_void,
    ) {
        let fault = unsafe { (*info).si_addr() as usize };
        let in_wasm_memory = ACTIVE
            .try_with(|active| {
                let mut scope = active.get();
                while !scope.is_null() {
                    let current = unsafe { &*scope };
                    if current.ranges.iter().any(|range| {
                        fault >= range.accessible_end
                            && fault >= range.start
                            && fault < range.reservation_end
                    }) {
                        return true;
                    }
                    scope = current.previous;
                }
                false
            })
            .unwrap_or(false);

        if in_wasm_memory && unsafe { (*info).si_code > 0 } {
            std::panic::resume_unwind(Box::new(MemoryFault(signal)));
        }

        let old = PREVIOUS.get().map(|actions| {
            if signal == libc::SIGSEGV {
                &actions.segv
            } else {
                &actions.bus
            }
        });
        unsafe {
            if let Some(old) = old {
                if old.sa_sigaction == libc::SIG_IGN && (*info).si_code <= 0 {
                    return;
                }
                if old.sa_sigaction != libc::SIG_DFL && old.sa_sigaction != libc::SIG_IGN {
                    if old.sa_flags & libc::SA_SIGINFO != 0 {
                        let handler: extern "C" fn(i32, *mut libc::siginfo_t, *mut c_void) =
                            core::mem::transmute(old.sa_sigaction);
                        handler(signal, info, context);
                    } else {
                        let handler: extern "C" fn(i32) = core::mem::transmute(old.sa_sigaction);
                        handler(signal);
                    }
                    return;
                }
            }
            // Returning to an unrelated synchronous fault would loop forever.
            libc::signal(signal, libc::SIG_DFL);
            libc::raise(signal);
            libc::_exit(128 + signal);
        }
    }

    pub(super) fn scope<R>(ranges: Vec<MemoryRange>, f: impl FnOnce() -> R) -> R {
        struct Guard(*const Scope);
        impl Drop for Guard {
            fn drop(&mut self) {
                ACTIVE.with(|active| active.set(self.0));
            }
        }
        let previous = ACTIVE.with(Cell::get);
        let scope = Scope { ranges, previous };
        ACTIVE.with(|active| active.set(&scope));
        let _guard = Guard(previous);
        f()
    }

    pub(super) fn catch<R>(f: impl FnOnce() -> Result<R>) -> Result<R> {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(result) => result,
            Err(payload) => match payload.downcast::<MemoryFault>() {
                Ok(fault) => {
                    let mut signals = unsafe { core::mem::zeroed() };
                    unsafe {
                        libc::sigemptyset(&mut signals);
                        libc::sigaddset(&mut signals, fault.0);
                        libc::pthread_sigmask(libc::SIG_UNBLOCK, &signals, ptr::null_mut());
                    }
                    Err(Error::Trap(crate::vm::TrapCode::MemoryOutOfBounds))
                }
                Err(payload) => std::panic::resume_unwind(payload),
            },
        }
    }
}

pub(crate) fn install() -> Result<()> {
    if cfg!(panic = "abort") {
        return Err(Error::Unsupported(
            "hardware memory checks require panic=unwind".into(),
        ));
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        platform::install()
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
    {
        Err(Error::Unsupported(
            "hardware memory checks require Linux x86-64 with glibc".into(),
        ))
    }
}

pub(crate) fn enabled() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        platform::enabled()
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
    {
        false
    }
}

pub(crate) fn scope<R>(ranges: Vec<MemoryRange>, f: impl FnOnce() -> R) -> R {
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        platform::scope(ranges, f)
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
    {
        let _ = ranges;
        f()
    }
}

pub(crate) fn catch<R>(f: impl FnOnce() -> Result<R>) -> Result<R> {
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        platform::catch(f)
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
    {
        f()
    }
}
