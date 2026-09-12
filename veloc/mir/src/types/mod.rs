//! Veloc MIR types, checked views, signatures and entity handles.

mod entities;
mod signature;
mod r#type;

pub use entities::*;
pub use signature::{CallConv, Signature};
pub use r#type::{CallableKind, ScalarType, Shape, Type, TypeBits, TypeSize, VectorType};
