//! Machine-facing IR shared by lowering, optimization and code generation.
//! Representation and decoding live here; target algorithms live in codegen.

#![no_std]
#![feature(const_trait_impl)]
extern crate alloc;

pub mod function;
pub mod instr;
pub mod module;
mod register;
pub mod symbol;

pub use function::*;
pub(crate) use instr::FieldPools;
pub use instr::*;
pub use module::*;
pub use register::*;
pub use symbol::*;
pub use veloc_bytecode::OperandRef;
pub use veloc_types::{Type, TypeBits, TypeInfo};

pub mod types {
    include!(concat!(env!("OUT_DIR"), "/types.rs"));
}
