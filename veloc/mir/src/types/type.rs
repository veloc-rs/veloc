//! Tagged MIR types, compact scalar/vector payloads, checked views and sizes.

use super::SigId;
use core::fmt;

include!(concat!(env!("OUT_DIR"), "/types.rs"));

pub use veloc_types::{CallableKind, Shape, TypeBits, TypeSize};

impl Type {
    /// Signatures belong to the containing module, just like function signatures.
    /// Keeping the tag outside the compact payload preserves fast scalar tests.
    pub const fn callable(signature: SigId, kind: CallableKind) -> Self {
        let tag = match kind {
            CallableKind::Local => 1u64,
            CallableKind::Owned => 2u64,
            CallableKind::Shared => 3u64,
        };
        Self((tag << 48) | ((signature.0 as u64) << 16))
    }

    pub const fn as_callable(self) -> Option<(SigId, CallableKind)> {
        if self.0 & 0xffff != 0 {
            return None;
        }
        let kind = match self.0 >> 48 {
            1 => CallableKind::Local,
            2 => CallableKind::Owned,
            3 => CallableKind::Shared,
            _ => return None,
        };
        Some((SigId((self.0 >> 16) as u32), kind))
    }

    pub const fn is_callable(self) -> bool {
        self.as_callable().is_some()
    }

    pub const fn is_owned(self) -> bool {
        matches!(self.as_callable(), Some((_, CallableKind::Owned)))
    }

    pub const fn is_compact(self) -> bool {
        self.0 <= u16::MAX as u64
    }
}

impl Type {
    // === 构造函数 ===

    /// Create a vector mask. Masks are boolean vectors rather than a separate
    /// type family.
    pub const fn new_mask(lanes: u16, scalable: bool) -> Option<Self> {
        let scalar = Self::BOOL.as_scalar().expect("BOOL is a scalar");
        let Some(vector) = scalar.vector(lanes, scalable) else {
            return None;
        };
        Some(vector.as_type())
    }

    // === 预定义常量 ===

    pub const INVALID: Self = Self(0);

    // === 访问器 ===

    /// Minimum lane count, treating a valid scalar as one lane.
    pub const fn lane_count(self) -> u16 {
        assert!(
            self.is_valid() && self.is_compact(),
            "type has no lane count"
        );
        1 << self.lanes_log2()
    }

    /// Whether the type encoding is valid. Structural signature references are
    /// checked against their module at the explicit validation phase.
    pub const fn is_valid(self) -> bool {
        self.is_callable() || (self.is_compact() && Self::from_raw(self.0 as u16).is_some())
    }

    /// Decode a raw value after validating all currently defined fields.
    pub const fn from_raw(raw: u16) -> Option<Self> {
        if raw == 0 || raw & !USED_MASK != 0 {
            return None;
        }
        let ty = Self(raw as u64);
        let Some(scalar) = ScalarType::from_code(ty.element_code()) else {
            return None;
        };
        let lanes_log2 = ty.lanes_log2();
        let scalable = ty.is_scalable();
        if (lanes_log2 == 0 && scalable) || (lanes_log2 > 0 && !scalar.can_vectorize()) {
            return None;
        }
        Some(ty)
    }
}

impl Default for Type {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Type {
    /// Decode storage into encoding-independent facts. No allocation or module lookup.
    #[inline]
    const fn facts(self) -> veloc_types::Type {
        if let Some((_, kind)) = self.as_callable() {
            return veloc_types::Type::callable(kind);
        }
        assert!(self.is_valid(), "invalid type has no concrete facts");
        let element = self.element_facts();
        if self.lanes_log2() == 0 {
            veloc_types::Type::scalar(element)
        } else {
            veloc_types::Type::vector(element, self.lane_count(), self.is_scalable())
                .expect("validated vector")
        }
    }

    #[inline]
    pub const fn element_bits(self) -> Option<u32> {
        self.facts().element_bits()
    }
    #[inline]
    pub fn bit_size(self) -> Option<TypeBits> {
        self.facts().bit_size()
    }
    #[inline]
    pub fn storage_size(self) -> TypeSize {
        self.facts().storage_size()
    }
    #[inline]
    pub fn fixed_size_bytes(self) -> Option<u32> {
        self.storage_size().fixed_bytes()
    }
    #[inline]
    pub fn min_size_bytes(self) -> Option<u32> {
        self.storage_size().min_bytes()
    }
    #[inline]
    pub fn min_bit_width(self) -> Option<u32> {
        self.bit_size().map(TypeBits::min_bits)
    }

    // Total query interfaces used by defs; scalar and callable inputs need no unchecked view.
    #[inline]
    pub const fn lanes(self) -> Option<u32> {
        self.facts().lanes()
    }
    #[inline]
    pub const fn shape(self) -> Option<Shape> {
        self.facts().shape()
    }
    #[inline]
    pub const fn is_fixed(self) -> bool {
        self.facts().is_fixed()
    }
    #[inline]
    pub const fn is_local(self) -> bool {
        self.facts().is_local()
    }
    #[inline]
    pub const fn is_shared(self) -> bool {
        self.facts().is_shared()
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_callable() {
            return fmt::Display::fmt(self, f);
        }
        if !self.is_valid() {
            return f.write_str("invalid");
        }
        if self.is_predicate() {
            let lanes = self.lane_count();
            if self.is_scalable() {
                write!(f, "<vscale x {} x mask>", lanes)
            } else {
                write!(f, "<{} x mask>", lanes)
            }
        } else if self.is_vector() {
            let elem = self.element_name(true);
            let lanes = self.lane_count();
            if self.is_scalable() {
                write!(f, "<vscale x {} x {}>", lanes, elem)
            } else {
                write!(f, "<{} x {}>", lanes, elem)
            }
        } else {
            f.write_str(self.element_name(true))
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((sig, kind)) = self.as_callable() {
            return write!(
                f,
                "{}<sig{}>",
                match kind {
                    CallableKind::Local => "local",
                    CallableKind::Owned => "owned",
                    CallableKind::Shared => "shared",
                },
                sig.0
            );
        }
        if !self.is_valid() {
            return f.write_str("invalid");
        }
        if self.is_predicate() {
            let lanes = self.lane_count();
            if self.is_scalable() {
                write!(f, "mask<scalable {}>", lanes)
            } else {
                write!(f, "mask<{}>", lanes)
            }
        } else if self.is_vector() {
            let elem = self.element_name(false);
            let lanes = self.lane_count();
            if self.is_scalable() {
                write!(f, "{}<scalable {}>", elem, lanes)
            } else {
                write!(f, "{}<{}>", elem, lanes)
            }
        } else {
            f.write_str(self.element_name(false))
        }
    }
}

/// A validated fixed or scalable vector Type, including boolean masks.
/// Construction requires a checked conversion:
///
/// ```compile_fail
/// let vector = veloc_mir::VectorType(veloc_mir::Type::I32);
/// ```
///
/// ```
/// use veloc_mir::Type;
/// let vector = Type::I32X4.as_vector().unwrap();
/// assert_eq!(vector.shape(), (4, false));
/// assert_eq!(vector.element_type().as_type(), Type::I32);
/// let ty: Type = vector.into();
/// assert_eq!(ty, Type::I32X4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VectorType(Type);

impl Type {
    /// Check validity and scalar shape once, then expose scalar-only operations.
    pub const fn as_scalar(self) -> Option<ScalarType> {
        // Only the scalar field may be set: reject vector/structural tags and
        // unused encoding bits before decoding. Never truncate an arbitrary Type.
        if self.0 & !(SCALAR_MASK as u64) != 0 {
            None
        } else {
            ScalarType::from_code(self.element_code())
        }
    }

    /// Check validity and vector shape once, then expose vector-only operations.
    pub const fn as_vector(self) -> Option<VectorType> {
        if self.is_compact() && self.is_valid() && self.lanes_log2() > 0 {
            Some(VectorType(self))
        } else {
            None
        }
    }
}

impl ScalarType {
    /// Form a vector. Pointers, invalid lane counts and unrepresentable shapes
    /// are rejected; the scalar receiver is already known to be valid.
    ///
    /// ```
    /// use veloc_mir::Type;
    /// let scalar = Type::I32.as_scalar().unwrap();
    /// let vector = scalar.vector(4, false).unwrap();
    /// assert_eq!(vector.as_type(), Type::I32X4);
    /// ```
    ///
    /// A scalar has no vector shape, and vector construction requires a scalar:
    ///
    /// ```compile_fail
    /// veloc_mir::ScalarType::I32.shape();
    /// ```
    ///
    /// ```compile_fail
    /// veloc_mir::Type::I32.vector(4, false);
    /// ```
    pub const fn vector(self, lanes: u16, scalable: bool) -> Option<VectorType> {
        if lanes < 2 || !lanes.is_power_of_two() || !self.can_vectorize() {
            return None;
        }
        let log2_lanes = lanes.trailing_zeros() as u16;
        if log2_lanes > LANES_LOG2_MAX {
            return None;
        }
        let scalable_bit = if scalable { SCALABLE_MASK } else { 0 };
        Some(VectorType(Type(
            self.as_type().0 | ((log2_lanes << LANES_LOG2_SHIFT) | scalable_bit) as u64,
        )))
    }
}

impl VectorType {
    pub const fn as_type(self) -> Type {
        self.0
    }

    pub const fn element_type(self) -> ScalarType {
        ScalarType::from_code(self.0.element_code()).expect("validated vector element")
    }

    /// Minimum lane count; scalable vectors have vscale times this many lanes.
    pub const fn lane_count(self) -> u16 {
        1 << self.0.lanes_log2()
    }

    pub const fn is_scalable(self) -> bool {
        self.0.is_scalable()
    }

    pub const fn is_fixed(self) -> bool {
        !self.is_scalable()
    }

    pub const fn shape(self) -> (u16, bool) {
        (self.lane_count(), self.is_scalable())
    }
}

impl From<ScalarType> for Type {
    fn from(ty: ScalarType) -> Self {
        ty.as_type()
    }
}

impl From<VectorType> for Type {
    fn from(ty: VectorType) -> Self {
        ty.as_type()
    }
}
