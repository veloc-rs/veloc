//! Runtime module management
//!
//! This module provides the runtime representation of compiled modules
//! and their associated metadata.

mod func_ref;
pub mod program;

use crate::bytecode::CompiledFunction;
use alloc::sync::Arc;
use cranelift_entity::PrimaryMap;
pub use func_ref::{CallTarget, FunctionRef};
pub use program::{Program, ProgramBuilder};
use veloc_ir::{FuncId, Module};

/// Runtime representation of a compiled module
pub(crate) struct RuntimeModule {
    /// The original IR module
    ir: Module,
    /// Compiled bytecode functions (None for imports)
    compiled: PrimaryMap<FuncId, Option<Arc<CompiledFunction>>>,
    /// Explicit target for every direct call slot.
    call_targets: PrimaryMap<FuncId, CallTarget>,
    /// Opaque function-reference handle for every defined function.
    func_refs: PrimaryMap<FuncId, Option<FunctionRef>>,
}
