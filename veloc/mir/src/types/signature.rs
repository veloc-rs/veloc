//! Function signatures and calling conventions.

use super::Type;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallConv {
    /// Standard System V ABI (e.g., for standard C functions on Linux)
    SystemV,
}

impl core::fmt::Display for CallConv {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CallConv::SystemV => write!(f, "system_v"),
        }
    }
}

/// A function signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    pub params: Vec<Type>,
    pub returns: Vec<Type>,
    pub call_conv: CallConv,
}

impl Signature {
    pub fn new(params: Vec<Type>, returns: Vec<Type>, call_conv: CallConv) -> Self {
        Self {
            params,
            returns,
            call_conv,
        }
    }
}
