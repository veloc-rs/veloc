#![feature(explicit_tail_calls)]
#![feature(macro_metavar_expr)]
#![feature(rust_preserve_none_cc)]
#![expect(incomplete_features)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
pub mod bytecode;
pub mod error;
pub mod host;
pub mod interpreter;
pub mod runtime;
pub mod value;

pub use bytecode::printer::FuncPrinter;
pub use error::{Error, Result};
pub use host::{HostFuncId, HostFunction};
pub use interpreter::{Interpreter, VirtualMemory};
pub use runtime::{CallTarget, FunctionRef, Program, ProgramBuilder};
pub use value::InterpreterValue;
pub use veloc_mir::ModuleId;
