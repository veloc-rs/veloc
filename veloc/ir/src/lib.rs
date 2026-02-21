#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod builder;
pub mod dfg;
pub mod error;
pub mod function;
pub mod inst;

pub mod layout;
pub mod module;
pub mod text;
pub mod types;
pub mod validator;

pub mod constant;
pub mod intrinsic;
mod opcode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallConv {
    /// Standard System V ABI (e.g., for standard C functions on Linux)
    SystemV,
}

impl core::fmt::Display for CallConv {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CallConv::SystemV => write!(f, "system_v"),
        }
    }
}

pub use builder::{FunctionBuilder, InstBuilder, ModuleBuilder};
pub use error::{Error, Result};
pub use function::Function;
pub use intrinsic::{Intrinsic, ids as intrinsic_ids};
pub use module::{Global, Linkage, Module, ModuleData};
pub use opcode::{FloatCC, IntCC, MemFlags, Opcode};
// Re-export text format module
pub use text::{
    fmt, parse_function, parse_module, fmt_type, parse_type_ir, 
    format as text_format, IRFormat, ParseError,
};
pub use types::{
    Block, BlockCall, FuncId, JumpTable, ScalarType, SigId, Signature, StackSlot, Type, Value,
    ValueDef, ValueList, Variable, VectorKind,
};

// Internal-only re-exports for the backend and passes
pub use dfg::DataFlowGraph;
pub use inst::{
    ConstantPoolData, ConstantPoolId, Inst, InstructionData, PtrIndexImm, PtrIndexImmId,
    VectorExtData, VectorExtId, VectorMemExtData, VectorMemExtId,
};
