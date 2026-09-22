// Shared with the generated-API fixture crate.
#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod builder;
pub mod constant;
pub mod dfg;
pub mod error;
pub mod function;
pub mod host;
pub mod inst;
pub mod intrinsic;
pub mod memory;
pub mod module;
pub mod text;
pub mod types;
pub mod validator;

pub use builder::{ModuleBuilder, SsaBuilder};
pub use constant::{ConstData, Constant, Float, Int, ScalarConst, VectorConst};
pub use error::{Error, Result};
pub use function::{EdgeRef, FuncBody, FuncDecl, FunctionRef, InstCursor};
pub use inst::type_methods;
pub use inst::{
    Arguments, FloatCC, Inst, InstView, InstWriter, IntCC, MemFlags, Opcode, Successor,
    SuccessorMut, Successors, VectorMemOptions,
};
pub use intrinsic::{Intrinsic, ids as intrinsic_ids};
pub use module::{Global, Linkage, Module, ModuleData};
pub use text::{ModuleParser, ParseError};
pub use types::{
    Block, BlockCall, CallConv, CallableKind, FuncId, ModuleId, ScalarType, SigId, Signature, Type,
    TypeBits, TypeInfo, TypeSize, Value, ValueDef, ValueList, Variable, VectorType,
};
