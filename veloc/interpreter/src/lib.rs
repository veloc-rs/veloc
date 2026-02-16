extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
pub mod bytecode;
pub mod error;
pub mod host;
pub mod interpreter;
pub mod value;

pub use error::{Error, Result};
pub use host::{HostFunc, ModuleId, Program};
pub use interpreter::{Interpreter, VirtualMemory};
pub use value::InterpreterValue;
