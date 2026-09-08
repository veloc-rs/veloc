#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod builder;
pub mod constant;
pub mod dfg;
pub mod error;
pub mod function;
pub mod inst;
pub mod intrinsic;
pub mod module;
pub mod text;
pub mod types;
pub mod validator;

pub use builder::{FunctionBuilder, InstBuilder, ModuleBuilder};
pub use error::{Error, Result};
pub use function::Function;
pub use inst::{
    Arguments, FloatCC, Inst, InstDraft, InstructionView, IntCC, MemFlags, Opcode, Successor,
    Successors, VectorMemOptions,
};
pub use intrinsic::{Intrinsic, ids as intrinsic_ids};
pub use module::{Global, Linkage, Module, ModuleData};
pub use text::{ModuleParser, ParseError};
pub use types::{
    Block, BlockCall, CallConv, FuncId, ModuleId, ScalarType, SigId, Signature, StackSlot, Type,
    TypeBits, TypeSize, Value, ValueDef, ValueList, Variable, VectorType,
};
