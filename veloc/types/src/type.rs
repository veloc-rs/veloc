//! Compact shared types, checked scalar/vector views and physical encoding.

use crate::{CallableKind, Scalar, Shape, SigId, TypeBits, TypeSize};
use core::fmt;

// Physical layout is a Rust implementation detail, not an OpSpec contract.
const SCALAR_MASK: u16 = 0x000f;
const LANES_LOG2_SHIFT: u32 = 4;
const LANES_LOG2_MASK: u16 = 0x00f0;
const SCALABLE_MASK: u16 = 0x0100;
const USED_MASK: u16 = SCALAR_MASK | LANES_LOG2_MASK | SCALABLE_MASK;
const LANES_LOG2_MAX: u16 = crate::MAX_VECTOR_LANES.trailing_zeros() as u16;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Type(u64);

// One Rust table owns scalar discriminants, names and their logical facts.
macro_rules! scalar_types {
    ($($name:ident = $code:literal => $fact:ident $(($bits:literal))?, $debug:literal, $text:literal;)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(u8)]
        pub enum ScalarType { $($name = $code,)* }
        impl ScalarType {
            #[inline]
            pub const fn from_element(element: Scalar) -> Option<Self> {
                match element { $($fact $(($bits))? => Some(Self::$name),)* _ => None }
            }
            #[inline]
            pub const fn code(self) -> u8 { self as u8 }
            #[inline]
            pub const fn from_code(code: u8) -> Option<Self> {
                match code { $($code => Some(Self::$name),)* _ => None }
            }
            #[inline]
            pub const fn as_type(self) -> Type { Type(self.code() as u64) }
            #[inline]
            pub const fn element(self) -> Scalar {
                match self { $(Self::$name => $fact $(($bits))?,)* }
            }
            #[inline]
            const fn can_vectorize(self) -> bool { !matches!(self, Self::PTR) }
            fn name(self, debug: bool) -> &'static str {
                match self { $(Self::$name => if debug { $debug } else { $text },)* }
            }
        }
        impl Type {
            $(pub const $name: Self = ScalarType::$name.as_type();)*
            pub fn from_name(name: &str) -> Option<Self> {
                match name { $($text => Some(Self::$name),)* _ => None }
            }
        }
    }
}
use crate::Scalar::{Bool, Float, Int, Ptr};
scalar_types! {
    I8 = 1 => Int(8), "I8", "i8";
    I16 = 2 => Int(16), "I16", "i16";
    I32 = 3 => Int(32), "I32", "i32";
    I64 = 4 => Int(64), "I64", "i64";
    F32 = 5 => Float(32), "F32", "f32";
    F64 = 6 => Float(64), "F64", "f64";
    BOOL = 7 => Bool, "Bool", "bool";
    PTR = 8 => Ptr, "Ptr", "ptr";
}
impl Type {
    #[inline]
    pub const fn from_scalar_code(code: u8) -> Option<Self> {
        match ScalarType::from_code(code) {
            Some(scalar) => Some(scalar.as_type()),
            None => None,
        }
    }
    #[inline]
    pub const fn to_raw(self) -> u16 {
        assert!(self.is_compact(), "structural type has no compact encoding");
        self.0 as u16
    }
    #[inline]
    const fn element_code(self) -> u8 {
        (self.0 & SCALAR_MASK as u64) as u8
    }
    #[inline]
    const fn lanes_log2(self) -> u16 {
        ((self.0 as u16) & LANES_LOG2_MASK) >> LANES_LOG2_SHIFT
    }
    #[inline]
    pub const fn is_scalable(self) -> bool {
        self.0 & SCALABLE_MASK as u64 != 0
    }
    /// Logical element kind; callers do not need to know physical discriminants.
    #[inline]
    pub const fn element(self) -> Option<crate::Scalar> {
        if self.is_compact() && self.is_valid() {
            Some(
                ScalarType::from_code(self.element_code())
                    .expect("validated scalar kind")
                    .element(),
            )
        } else {
            None
        }
    }
    fn element_name(self, debug: bool) -> &'static str {
        ScalarType::from_code(self.element_code())
            .expect("validated scalar kind")
            .name(debug)
    }
    #[inline]
    pub const fn is_scalar(self) -> bool {
        self.as_scalar().is_some()
    }
    #[inline]
    pub const fn is_vector(self) -> bool {
        self.as_vector().is_some()
    }
    #[inline]
    pub const fn is_integer(self) -> bool {
        matches!(self.element(), Some(Int(_)))
    }
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self.element(), Some(Float(_)))
    }
    #[inline]
    pub const fn is_ptr(self) -> bool {
        matches!(self.as_scalar(), Some(ScalarType::PTR))
    }
    #[inline]
    pub const fn is_predicate(self) -> bool {
        self.is_vector() && matches!(self.element(), Some(Bool))
    }
}

impl Type {
    /// The signature ID belongs to the owning context, not to a global registry.
    /// Keeping the tag outside the compact payload preserves fast scalar tests.
    #[inline]
    pub const fn callable(signature: SigId, kind: CallableKind) -> Self {
        let tag = match kind {
            CallableKind::Local => 1u64,
            CallableKind::Owned => 2u64,
            CallableKind::Shared => 3u64,
        };
        Self((tag << 48) | ((signature.0 as u64) << 16))
    }

    #[inline]
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

    #[inline]
    pub const fn is_callable(self) -> bool {
        self.as_callable().is_some()
    }

    #[inline]
    pub const fn is_owned(self) -> bool {
        matches!(self.as_callable(), Some((_, CallableKind::Owned)))
    }

    #[inline]
    pub const fn is_compact(self) -> bool {
        self.0 <= u16::MAX as u64
    }
}

impl Type {
    /// Create a vector mask. Masks are boolean vectors rather than a separate
    /// type family.
    #[inline]
    pub const fn new_mask(lanes: u16, scalable: bool) -> Option<Self> {
        let Some(vector) = ScalarType::BOOL.vector(lanes, scalable) else {
            return None;
        };
        Some(vector.as_type())
    }

    pub const INVALID: Self = Self(0);

    /// Minimum lane count, treating a valid scalar as one lane.
    #[inline]
    pub const fn lane_count(self) -> u16 {
        assert!(
            self.is_valid() && self.is_compact(),
            "type has no lane count"
        );
        1 << self.lanes_log2()
    }

    /// Whether the type encoding is valid. Structural signature references are
    /// checked against their module at the explicit validation phase.
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.is_callable() || (self.is_compact() && Self::from_raw(self.0 as u16).is_some())
    }

    /// Decode a raw value after validating all currently defined fields.
    #[inline]
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
    #[inline]
    pub const fn element_bits(self) -> Option<u32> {
        match self.element() {
            Some(element) => element.element_bits(),
            None => None,
        }
    }
    #[inline]
    pub const fn lanes(self) -> Option<u32> {
        if self.is_compact() && self.is_valid() {
            Some(self.lane_count() as u32)
        } else {
            None
        }
    }
    #[inline]
    pub const fn shape(self) -> Option<Shape> {
        match self.as_vector() {
            Some(vector) => Some(vector.shape()),
            None => None,
        }
    }
    #[inline]
    pub const fn is_fixed(self) -> bool {
        self.is_vector() && !self.is_scalable()
    }
    #[inline]
    pub const fn is_local(self) -> bool {
        matches!(self.as_callable(), Some((_, CallableKind::Local)))
    }
    #[inline]
    pub const fn is_shared(self) -> bool {
        matches!(self.as_callable(), Some((_, CallableKind::Shared)))
    }
    #[inline]
    pub fn bit_size(self) -> Option<TypeBits> {
        let bits = self.element_bits()?.checked_mul(self.lanes()?)?;
        Some(if self.is_scalable() {
            TypeBits::Scalable { min_bits: bits }
        } else {
            TypeBits::Fixed(bits)
        })
    }
    /// Byte-addressed representation, not a target ABI or packed predicate layout.
    #[inline]
    pub fn storage_size(self) -> TypeSize {
        let Some(bytes) = self
            .element_bits()
            .and_then(|bits| bits.div_ceil(8).checked_mul(self.lanes()?))
        else {
            return TypeSize::TargetDependent;
        };
        if self.is_scalable() {
            TypeSize::Scalable { min_bytes: bytes }
        } else {
            TypeSize::Fixed(bytes)
        }
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
/// let vector = veloc_types::VectorType(veloc_types::Type::I32);
/// ```
///
/// ```
/// use veloc_types::{Type, ScalarType};
/// let vector = ScalarType::I32.vector(4, false).unwrap().as_type().as_vector().unwrap();
/// assert_eq!(vector.shape(), (4, false));
/// assert_eq!(vector.element_type().as_type(), Type::I32);
/// let ty: Type = vector.into();
/// assert_eq!(ty, ScalarType::I32.vector(4, false).unwrap().as_type());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VectorType(Type);

impl Type {
    /// Check validity and scalar shape once, then expose scalar-only operations.
    #[inline]
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
    #[inline]
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
    /// use veloc_types::{Type, ScalarType};
    /// let scalar = Type::I32.as_scalar().unwrap();
    /// let vector = scalar.vector(4, false).unwrap();
    /// assert_eq!(vector.as_type(), ScalarType::I32.vector(4, false).unwrap().as_type());
    /// ```
    ///
    /// A scalar has no vector shape, and vector construction requires a scalar:
    ///
    /// ```compile_fail
    /// veloc_types::ScalarType::I32.shape();
    /// ```
    ///
    /// ```compile_fail
    /// veloc_types::Type::I32.vector(4, false);
    /// ```
    #[inline]
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
    #[inline]
    pub const fn as_type(self) -> Type {
        self.0
    }

    #[inline]
    pub const fn element_type(self) -> ScalarType {
        ScalarType::from_code(self.0.element_code()).expect("validated vector element")
    }

    /// Minimum lane count; scalable vectors have vscale times this many lanes.
    #[inline]
    pub const fn lane_count(self) -> u16 {
        1 << self.0.lanes_log2()
    }

    #[inline]
    pub const fn is_scalable(self) -> bool {
        self.0.is_scalable()
    }

    #[inline]
    pub const fn is_fixed(self) -> bool {
        !self.is_scalable()
    }

    #[inline]
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
