#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub use veloc_backend as backend;
pub use veloc_interpreter as interpreter;
pub use veloc_ir as ir;

pub use ir::{Error, Result};
