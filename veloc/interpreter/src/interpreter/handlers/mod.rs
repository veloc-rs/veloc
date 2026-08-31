//! Opcode semantics grouped by domain. The parent module owns dispatch plumbing.

mod constants;
mod control;
mod conversion;
mod float;
mod integer;
mod intrinsic;
mod memory;

pub(crate) use constants::*;
pub(crate) use control::*;
pub(crate) use conversion::*;
pub(crate) use float::*;
pub(crate) use integer::*;
pub(crate) use memory::*;
