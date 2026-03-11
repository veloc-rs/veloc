mod compile;
mod inst;
pub mod printer;

pub use compile::CompiledFunction;
pub(crate) use compile::{DataSection, JumpTarget};

pub(crate) use compile::compile_function;
pub(crate) use inst::{DecodedInstruction, Instruction, Opcode, Reg};
