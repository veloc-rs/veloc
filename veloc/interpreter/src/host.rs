//! Host function support
//!
//! This module provides functionality for registering and calling
//! host (native) functions from the interpreter.

use crate::value::InterpreterValue;
use alloc::sync::Arc;
use cranelift_entity::entity_impl;
use veloc_mir::Signature;

/// A host callback and the exact signature it accepts.
#[derive(Clone)]
pub struct HostFunction {
    callback: Arc<dyn Fn(&mut [InterpreterValue]) + Send + Sync>,
    signature: Signature,
}

impl core::fmt::Debug for HostFunction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostFunction")
            .field("signature", &self.signature)
            .finish_non_exhaustive()
    }
}

impl HostFunction {
    pub fn new<F>(signature: Signature, callback: F) -> Self
    where
        F: Fn(&mut [InterpreterValue]) + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
            signature,
        }
    }

    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    pub(crate) fn call(
        &self,
        values: &mut [InterpreterValue],
        args: usize,
        results: usize,
    ) -> crate::Result<()> {
        if args != self.signature.params.len()
            || results != self.signature.returns.len()
            || values.len() < args.max(results).max(1)
        {
            return Err(crate::Error::InvalidHostCall);
        }
        (self.callback)(values);
        Ok(())
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
