//! Machine-facing IR shared by lowering, optimization and code generation.
//! Representation and decoding live here; target algorithms live in codegen.

#![no_std]
extern crate alloc;

pub mod error;
pub mod extra;
pub mod function;
pub mod instr;
pub mod module;
pub mod stages;
pub mod symbol;
pub mod use_def;

pub use error::{DecodeError, Result};
pub use extra::*;
pub use function::*;
pub use instr::*;
pub use module::*;
pub use symbol::*;
pub use use_def::UseDefChain;
pub use veloc_mir::Value as ValueId;
