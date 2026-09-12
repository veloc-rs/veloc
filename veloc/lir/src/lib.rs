//! Machine-facing IR shared by lowering, optimization and code generation.
//! Representation and decoding live here; target algorithms live in codegen.

#![no_std]
extern crate alloc;

pub mod error;
pub mod extra;
pub mod function;
pub mod instr;
pub mod memory;
pub mod module;
pub mod stages;
pub mod symbol;
pub mod use_def;
mod validation;

pub use error::{DecodeError, Result};
pub use extra::*;
pub use function::*;
pub use instr::*;
pub use memory::*;
pub use module::*;
pub use symbol::*;
pub use use_def::UseDefChain;
pub use validation::TypeError;
pub use veloc_mir::Value as ValueId;
pub use veloc_types::{Type, TypeBits};

pub mod types {
    include!(concat!(env!("OUT_DIR"), "/types.rs"));
}
