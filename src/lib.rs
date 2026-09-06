#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub use veloc_codegen as codegen;
pub use veloc_interpreter as interpreter;
pub use veloc_mir as mir;

pub use mir::{Error, Result};
