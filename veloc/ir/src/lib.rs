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
pub mod opspec;
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
pub use text::{ModuleParser, ParseError};
pub use types::{
    Block, BlockCall, FuncId, JumpTable, ModuleId, ScalarType, SigId, Signature, StackSlot, Type,
    TypeSize, Value, ValueDef, ValueList, Variable,
};

pub use inst::{Inst, InstructionData, VectorMemOptions};
