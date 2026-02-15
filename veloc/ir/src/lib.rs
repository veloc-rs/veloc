#[cfg(feature = "std")]
extern crate std;

#[macro_use]
extern crate alloc;

pub mod builder;
pub mod dfg;
pub mod error;
pub mod function;
pub mod inst;

pub mod layout;
pub mod module;
pub mod passes;
pub mod printer;
pub mod types;
pub mod validator;

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
pub use inst::{FloatCC, IntCC, MemFlags};
pub use module::{Global, Linkage, Module, ModuleData};
pub use types::{
    Block, BlockCall, FuncId, JumpTable, SigId, Signature, StackSlot, Type, Value, ValueList,
    Variable,
};

// Internal-only re-exports for the backend and passes
pub use dfg::DataFlowGraph;
pub use inst::{InstructionData, Opcode};
pub use types::Inst;
