//! Veloc MIR types, checked views, signatures and entity handles.

mod entities;

pub use entities::*;
pub use veloc_types::{CallConv, SigId, Signature};
pub use veloc_types::{CallableKind, ScalarType, Shape, Type, TypeBits, TypeSize, VectorType};

include!(concat!(env!("OUT_DIR"), "/types.rs"));
