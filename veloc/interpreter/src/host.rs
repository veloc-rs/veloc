//! Host function support
//!
//! This module provides functionality for registering and calling
//! host (native) functions from the interpreter.

use crate::value::InterpreterValue;
use ::alloc::sync::Arc;
use cranelift_entity::entity_impl;

/// Type alias for host function closures
pub type HostFunction = Arc<dyn Fn(&[InterpreterValue]) -> InterpreterValue + Send + Sync>;

/// Trampoline function type for calling host functions
pub type TrampolineFn = unsafe extern "C" fn(
    env: *mut u8,
    args_results: *mut InterpreterValue,
    param_count: usize,
    buffer_len: usize,
);

/// Internal representation of a host function
pub struct HostFunctionInner {
    pub(crate) handler: TrampolineFn,
    pub(crate) env: *mut u8,
    pub(crate) drop_fn: fn(*mut u8),
}

unsafe impl Send for HostFunctionInner {}
unsafe impl Sync for HostFunctionInner {}

impl Drop for HostFunctionInner {
    fn drop(&mut self) {
        (self.drop_fn)(self.env);
    }
}

/// A callable host function handle
#[derive(Clone)]
pub struct HostFunc(pub Arc<HostFunctionInner>);

impl core::fmt::Debug for HostFunc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostFunc").finish()
    }
}

impl HostFunc {
    /// Call the host function with the given arguments
    pub fn call(&self, args_results: &mut [InterpreterValue], param_count: usize) {
        unsafe {
            (self.0.handler)(
                self.0.env,
                args_results.as_mut_ptr(),
                param_count,
                args_results.len(),
            );
        }
    }
}

/// A reference to a host function identifier
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostFuncId(u32);
entity_impl!(HostFuncId);

impl core::fmt::Debug for HostFuncId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "host{}", self.0)
    }
}
