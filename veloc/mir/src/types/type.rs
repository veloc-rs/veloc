//! Compact MIR types, checked views, sizes and text representation.

use core::fmt;

include!(concat!(env!("OUT_DIR"), "/encoding.rs"));
include!(concat!(env!("OUT_DIR"), "/scalars.rs"));

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
        assert!(self.is_valid(), "invalid MIR type has no lane count");
        1 << self.lanes_log2()
    }

    /// Whether this value has a valid compact type encoding.
    pub const fn is_valid(self) -> bool {
        Self::from_raw(self.0).is_some()
    }

    /// Decode a raw value after validating all currently defined fields.
    pub const fn from_raw(raw: u16) -> Option<Self> {
        if raw == 0 || raw & !USED_MASK != 0 {
            return None;
        }
        let ty = Self(raw);
        let Some(scalar) = Self::from_scalar_code(ty.element_code()) else {
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

/// Storage size in bytes in the MIR's byte representation.
/// This is separate from logical bit width and target register layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSize {
    Fixed(u32),
    Scalable { min_bytes: u32 },
    TargetDependent,
}

impl TypeSize {
    /// Return the exact size when it is independent of the target and runtime.
    pub const fn fixed_bytes(self) -> Option<u32> {
        match self {
            Self::Fixed(bytes) => Some(bytes),
            Self::Scalable { .. } | Self::TargetDependent => None,
        }
    }

    /// Return the statically known minimum size, if one exists.
    pub const fn min_bytes(self) -> Option<u32> {
        match self {
            Self::Fixed(bytes) | Self::Scalable { min_bytes: bytes } => Some(bytes),
            Self::TargetDependent => None,
        }
    }
}

/// Logical size in bits, preserving a vector's runtime scale factor.
///
/// `Scalable { min_bits: n }` denotes `vscale * n` bits, where the same positive
/// runtime `vscale` applies throughout an execution. It is not equal to
/// `Fixed(n)`, even though both have the same minimum size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeBits {
    Fixed(u32),
    Scalable { min_bits: u32 },
}

impl TypeBits {
    pub const fn fixed_bits(self) -> Option<u32> {
        match self {
            Self::Fixed(bits) => Some(bits),
            Self::Scalable { .. } => None,
        }
    }

    pub const fn min_bits(self) -> u32 {
        match self {
            Self::Fixed(bits) | Self::Scalable { min_bits: bits } => bits,
        }
    }
}

impl Type {
    /// Logical size of the whole value, including its runtime scale factor.
    /// Pointers have no target-independent bit size.
    pub fn bit_size(self) -> Option<TypeBits> {
        let min_bits = self.element_bits()? * u32::from(self.lane_count());
        Some(if self.is_scalable() {
            TypeBits::Scalable { min_bits }
        } else {
            TypeBits::Fixed(min_bits)
        })
    }

    /// Storage size of the value in the MIR's byte representation.
    /// Boolean vectors use one byte per lane, not packed logical bits.
    pub fn storage_size(self) -> TypeSize {
        let Some(lane_bits) = self.element_bits() else {
            return TypeSize::TargetDependent;
        };
        let min_bytes = lane_bits.div_ceil(8) * u32::from(self.lane_count());
        if self.is_scalable() {
            TypeSize::Scalable { min_bytes }
        } else {
            TypeSize::Fixed(min_bytes)
        }
    }

    pub fn fixed_size_bytes(self) -> Option<u32> {
        self.storage_size().fixed_bytes()
    }

    pub fn min_size_bytes(self) -> Option<u32> {
        self.storage_size().min_bytes()
    }

    /// Minimum logical bit width. Equal minima do not imply equal sizes;
    /// use [`Self::bit_size`] when checking bitcast compatibility.
    pub fn min_bit_width(self) -> Option<u32> {
        self.bit_size().map(TypeBits::min_bits)
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

pub(crate) fn parse_type(s: &str) -> Option<Type> {
    let Some((base, shape)) = s.split_once('<') else {
        return Type::from_name(s);
    };
    let shape = shape.strip_suffix('>')?;
    let scalar = if base == "mask" {
        Type::BOOL
    } else {
        Type::from_name(base)?
    };
    let (scalable, lanes) = match shape.strip_prefix("scalable ") {
        Some(lanes) => (true, lanes),
        None => (false, shape),
    };
    Some(
        scalar
            .as_scalar()?
            .vector(lanes.parse().ok()?, scalable)?
            .as_type(),
    )
}

/// A validated scalar Type, including pointers. This is a view, not another
/// type encoding. A scalar has no vector shape:
///
/// ```compile_fail
/// let scalar = veloc_mir::Type::I32.as_scalar().unwrap();
/// scalar.shape();
/// ```
///
/// Vector construction requires a scalar view, not an arbitrary Type:
///
/// ```compile_fail
/// veloc_mir::Type::I32.vector(4, false);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ScalarType(Type);

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
        // Use structural shape, not a user-declared predicate's type set.
        if self.is_valid() && self.lanes_log2() == 0 {
            Some(ScalarType(self))
        } else {
            None
        }
    }

    /// Check validity and vector shape once, then expose vector-only operations.
    pub const fn as_vector(self) -> Option<VectorType> {
        if self.is_valid() && self.lanes_log2() > 0 {
            Some(VectorType(self))
        } else {
            None
        }
    }
}

impl ScalarType {
    pub const fn as_type(self) -> Type {
        self.0
    }

    /// Layout-independent scalar code for compact backend metadata.
    pub const fn code(self) -> u8 {
        self.0.element_code()
    }

    /// Form a vector. Pointers, invalid lane counts and unrepresentable shapes
    /// are rejected; the scalar receiver is already known to be valid.
    ///
    /// ```
    /// use veloc_mir::Type;
    /// let scalar = Type::I32.as_scalar().unwrap();
    /// let vector = scalar.vector(4, false).unwrap();
    /// assert_eq!(vector.as_type(), Type::I32X4);
    /// ```
    pub const fn vector(self, lanes: u16, scalable: bool) -> Option<VectorType> {
        if lanes < 2 || !lanes.is_power_of_two() || !self.0.can_vectorize() {
            return None;
        }
        let log2_lanes = lanes.trailing_zeros() as u16;
        if log2_lanes > LANES_LOG2_MAX {
            return None;
        }
        let scalable_bit = if scalable { SCALABLE_MASK } else { 0 };
        Some(VectorType(Type(
            self.0.0 | (log2_lanes << LANES_LOG2_SHIFT) | scalable_bit,
        )))
    }
}

impl VectorType {
    pub const fn as_type(self) -> Type {
        self.0
    }

    pub const fn element_type(self) -> ScalarType {
        ScalarType(self.0.element_type())
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
