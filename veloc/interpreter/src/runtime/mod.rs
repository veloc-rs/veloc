//! Runtime module management
//!
//! This module provides the runtime representation of compiled modules
//! and their associated metadata.

pub mod program;
pub mod ptr;

use crate::bytecode::CompiledFunction;
use ::alloc::sync::Arc;
use ::alloc::vec::Vec;
use cranelift_entity::SecondaryMap;
pub use program::Program;
pub use ptr::{ImportTarget, VMFuncPointer};
use veloc_ir::{FuncId, Module};

/// Runtime representation of a compiled module
pub(crate) struct RuntimeModule {
    /// The original IR module
    ir: Module,
    /// Compiled bytecode functions (None for imports)
    pub(crate) compiled: SecondaryMap<FuncId, Option<Arc<CompiledFunction>>>,
    /// Resolved import links for each function
    pub(crate) links: SecondaryMap<FuncId, ImportTarget>,
}

impl RuntimeModule {
    /// Create a new runtime module
    pub fn new(
        ir: Module,
        compiled: SecondaryMap<FuncId, Option<Arc<CompiledFunction>>>,
        links: SecondaryMap<FuncId, ImportTarget>,
    ) -> Self {
        Self {
            ir,
            compiled,
            links,
        }
    }

    /// Get the original IR module
    pub fn ir(&self) -> &Module {
        &self.ir
    }
}
