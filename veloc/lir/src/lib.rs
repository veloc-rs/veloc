//! Machine-facing IR shared by lowering, optimization and code generation.
//! Representation and decoding live here; target algorithms live in codegen.

#![no_std]
#![feature(const_trait_impl)]
extern crate alloc;

pub mod error;
pub mod extra;
pub mod function;
pub mod instr;
pub mod memory;
pub mod module;
mod store;
pub mod symbol;
pub mod use_def;
mod validation;

pub use error::{Result, ValidationError};
pub use extra::*;
pub use function::*;
pub use instr::*;
pub use memory::*;
pub use module::*;
pub use store::{InstStore, InstWriter, RegEffects};
pub use symbol::*;
pub use use_def::{RefLocation, RefRole, RegRef, RegRefs};
pub use validation::TypeError;
pub use veloc_mir::Value as ValueId;
pub use veloc_types::{Type, TypeBits, TypeInfo};

pub mod types {
    include!(concat!(env!("OUT_DIR"), "/types.rs"));
}
