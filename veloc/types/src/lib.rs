//! Shared types, compact representation and interned signature storage.
//! No dependency on MIR entities, instruction containers or code generation.
#![no_std]

extern crate alloc;
mod signature;
pub use signature::{CallConv, SigId, Signature, SignatureError, Signatures};

/// Callable environment contracts, not a CPS calling convention.
/// Lifetime and call multiplicity are distinct; these are the combinations
/// supported by the IR, not a claim that every owned closure must be one-shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallableKind {
    /// Borrows the creating activation; reusable while that activation is alive.
    Local,
    /// Owns captures; must be called, transferred or explicitly dropped once.
    Owned,
    /// Reentrant immutable environment containing only duplicable captures.
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scalar {
    Int(u32),
    Float(u32),
    Bool,
    Ptr,
}

impl Scalar {
    #[inline]
    pub const fn element_bits(self) -> Option<u32> {
        match self {
            Self::Int(bits) | Self::Float(bits) => Some(bits),
            Self::Bool => Some(1),
            Self::Ptr => None,
        }
    }
}

/// Minimum lanes and whether they are multiplied by runtime vscale.
pub type Shape = (u16, bool);

/// Maximum representable minimum lane count.
pub const MAX_VECTOR_LANES: u16 = 1 << 15;

mod r#type;
pub use r#type::{ScalarType, Type, VectorType};

/// Storage size in the IR's byte-addressed representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSize {
    Fixed(u32),
    Scalable { min_bytes: u32 },
    TargetDependent,
}

impl TypeSize {
    #[inline]
    pub const fn fixed_bytes(self) -> Option<u32> {
        match self {
            Self::Fixed(bytes) => Some(bytes),
            _ => None,
        }
    }
    #[inline]
    pub const fn min_bytes(self) -> Option<u32> {
        match self {
            Self::Fixed(bytes) | Self::Scalable { min_bytes: bytes } => Some(bytes),
            Self::TargetDependent => None,
        }
    }
}

/// Logical bits, preserving runtime scale. Equal minima do not imply equal sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeBits {
    Fixed(u32),
    Scalable { min_bits: u32 },
}

impl TypeBits {
    #[inline]
    pub const fn fixed_bits(self) -> Option<u32> {
        match self {
            Self::Fixed(bits) => Some(bits),
            _ => None,
        }
    }
    #[inline]
    pub const fn min_bits(self) -> u32 {
        match self {
            Self::Fixed(bits) | Self::Scalable { min_bits: bits } => bits,
        }
    }
}
