//! Machine-facing IR shared by lowering, optimization and code generation.
//! Representation and decoding live here; target algorithms live in codegen.

#![no_std]
#![feature(const_trait_impl)]
extern crate alloc;

pub mod control;
mod regmask;
pub use regmask::RegMask;
pub mod error;
mod fields;
pub mod function;
pub mod instr;
pub mod layout;
pub mod memory;
pub mod module;
mod store;
pub mod symbol;
pub mod use_def;
mod validation;

pub use control::*;
pub use error::{Result, ValidationError};
pub use fields::FieldView;
pub(crate) use fields::{FieldPools, Fields};
pub use function::*;
pub use instr::*;
pub use memory::*;
pub use module::*;
pub use store::{InstEditor, InstStore, InstWriter, RegEffects};
pub use symbol::*;
pub use use_def::{OperandId, RefRole, RegRef, RegRefs};
pub use validation::TypeError;
pub use veloc_types::{Type, TypeBits, TypeInfo};

pub mod types {
    include!(concat!(env!("OUT_DIR"), "/types.rs"));
}
