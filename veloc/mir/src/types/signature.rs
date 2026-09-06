//! Function signatures.

use super::Type;
use alloc::vec::Vec;

/// A function signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    pub params: alloc::vec::Vec<Type>,
    pub returns: alloc::vec::Vec<Type>,
    pub call_conv: crate::CallConv,
}

impl Signature {
    pub fn new(params: Vec<Type>, returns: Vec<Type>, call_conv: crate::CallConv) -> Self {
        Self {
            params,
            returns,
            call_conv,
        }
    }
}
