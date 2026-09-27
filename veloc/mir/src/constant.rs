//! Exact constants. Scalar views add guarantees, not another representation tag.
use crate::{ScalarType, Type, VectorType};
use alloc::sync::Arc;
use veloc_types::TypeInfo;

/// A target-independent scalar bit pattern. Pointer constants are not modeled.
/// Unused high bits are always zero; equality preserves NaN payloads and -0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarConst {
    ty: ScalarType,
    bits: u64,
}

impl ScalarConst {
    /// Check that the type is supported and the bit pattern fits exactly.
    pub fn from_bits(ty: Type, bits: u64) -> Option<Self> {
        let scalar = ty.as_scalar()?;
        let width = ty.element_bits()?;
        if width > 64 || (width < 64 && bits >> width != 0) {
            return None;
        }
        Some(Self { ty: scalar, bits })
    }
    pub const fn ty(self) -> Type {
        self.ty.as_type()
    }
    pub const fn to_bits(self) -> u64 {
        self.bits
    }
    pub fn as_int(self) -> Option<Int> {
        self.ty().is_integer().then_some(Int(self))
    }
    pub fn as_float(self) -> Option<Float> {
        self.ty().is_float().then_some(Float(self))
    }
    pub fn as_bool(self) -> Option<bool> {
        (self.ty() == Type::BOOL).then_some(self.bits != 0)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Int(ScalarConst);

impl Int {
    /// Integer arithmetic is modulo the declared width: discard unused high bits.
    pub fn from_bits(ty: Type, bits: u64) -> Option<Self> {
        if !ty.is_integer() || ty.as_scalar().is_none() {
            return None;
        }
        let width = ty.element_bits()?;
        if width > 64 {
            return None;
        }
        let bits = bits & (u64::MAX >> (64 - width));
        ScalarConst::from_bits(ty, bits).map(Self)
    }
    pub const fn ty(self) -> Type {
        self.0.ty()
    }
    pub const fn to_bits(self) -> u64 {
        self.0.bits
    }
    pub fn signed(self) -> i64 {
        let shift = 64 - self.ty().element_bits().expect("integer width");
        ((self.0.bits << shift) as i64) >> shift
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Float(ScalarConst);

impl Float {
    pub const fn from_f32_bits(bits: u32) -> Self {
        Self(ScalarConst {
            ty: ScalarType::F32,
            bits: bits as u64,
        })
    }
    pub const fn from_f64_bits(bits: u64) -> Self {
        Self(ScalarConst {
            ty: ScalarType::F64,
            bits,
        })
    }
    pub const fn ty(self) -> Type {
        self.0.ty()
    }
    pub const fn to_bits(self) -> u64 {
        self.0.bits
    }
    pub fn as_f32(self) -> Option<f32> {
        (self.ty() == Type::F32).then(|| f32::from_bits(self.to_bits() as u32))
    }
    pub fn as_f64(self) -> Option<f64> {
        (self.ty() == Type::F64).then(|| f64::from_bits(self.to_bits()))
    }
}

impl From<Int> for ScalarConst {
    fn from(value: Int) -> Self {
        value.0
    }
}
impl From<Float> for ScalarConst {
    fn from(value: Float) -> Self {
        value.0
    }
}
impl TryFrom<ScalarConst> for Int {
    type Error = &'static str;
    fn try_from(value: ScalarConst) -> Result<Self, Self::Error> {
        value.as_int().ok_or("expected an integer constant")
    }
}
impl TryFrom<ScalarConst> for Float {
    type Error = &'static str;
    fn try_from(value: ScalarConst) -> Result<Self, Self::Error> {
        value.as_float().ok_or("expected a floating constant")
    }
}

macro_rules! integer_from {
    ($($rust:ty => $ty:ident),* $(,)?) => {$(
        impl From<$rust> for Int {
            fn from(value: $rust) -> Self {
                Self::from_bits(Type::$ty, value as u64).unwrap()
            }
        }
        impl From<$rust> for ScalarConst {
            fn from(value: $rust) -> Self { Int::from(value).into() }
        }
    )*};
}
integer_from!(i8 => I8, u8 => I8, i16 => I16, u16 => I16, i32 => I32, u32 => I32, i64 => I64, u64 => I64);

impl From<f32> for Float {
    fn from(value: f32) -> Self {
        Self::from_f32_bits(value.to_bits())
    }
}
impl From<f64> for Float {
    fn from(value: f64) -> Self {
        Self::from_f64_bits(value.to_bits())
    }
}
impl From<f32> for ScalarConst {
    fn from(value: f32) -> Self {
        Float::from(value).into()
    }
}
impl From<f64> for ScalarConst {
    fn from(value: f64) -> Self {
        Float::from(value).into()
    }
}
impl From<bool> for ScalarConst {
    fn from(value: bool) -> Self {
        Self {
            ty: ScalarType::BOOL,
            bits: value as u64,
        }
    }
}

/// Vector payloads own their immutable contents. Cloning shares dense bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstData {
    Dense(Arc<[u8]>),
    Splat(u64),
}

/// Exact, self-contained literals; no function-local payload handles.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    Scalar(ScalarConst),
    Vector(VectorConst),
}

impl Constant {
    pub const fn ty(&self) -> Type {
        match self {
            Self::Scalar(value) => value.ty(),
            Self::Vector(value) => value.ty(),
        }
    }
    pub fn as_scalar(&self) -> Option<ScalarConst> {
        match self {
            Self::Scalar(value) => Some(*value),
            Self::Vector(_) => None,
        }
    }
    pub fn as_vector(&self) -> Option<&VectorConst> {
        match self {
            Self::Vector(value) => Some(value),
            Self::Scalar(_) => None,
        }
    }
}

/// Vector construction fixes the type; payload validity is checked by validation.
/// Read through &VectorConst to avoid cloning the shared byte allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorConst {
    ty: VectorType,
    data: ConstData,
}

impl VectorConst {
    pub fn dense(ty: VectorType, bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            ty,
            data: ConstData::Dense(bytes.into()),
        }
    }

    pub fn splat(value: ScalarConst, lanes: u16, scalable: bool) -> Option<Self> {
        Some(Self {
            ty: value.ty.vector(lanes, scalable)?,
            data: ConstData::Splat(value.bits),
        })
    }

    pub const fn ty(&self) -> Type {
        self.ty.as_type()
    }
    pub const fn data(&self) -> &ConstData {
        &self.data
    }

    pub fn splat_value(&self) -> Option<ScalarConst> {
        let ConstData::Splat(bits) = self.data() else {
            return None;
        };
        Some(ScalarConst {
            ty: self.ty.element_type(),
            bits: *bits,
        })
    }
}

impl crate::type_methods::VectorConstInfo for VectorConst {
    fn bytes(&self) -> Option<&[u8]> {
        match &self.data {
            ConstData::Dense(bytes) => Some(bytes),
            ConstData::Splat(_) => None,
        }
    }

    /// Maximum unsigned integer lane, without allocating decoded lanes.
    fn unsigned_max(&self) -> Option<u64> {
        let element = self.ty.element_type();
        if !element.as_type().is_integer() {
            return None;
        }
        match &self.data {
            ConstData::Splat(bits) => Some(*bits),
            ConstData::Dense(bytes) => {
                if bytes.len() != self.encoded_size()? as usize {
                    return None;
                }
                let width = element.as_type().element_bits()? as usize / 8;
                bytes
                    .chunks_exact(width)
                    .map(|lane| {
                        let mut bits = [0; 8];
                        bits[..width].copy_from_slice(lane);
                        u64::from_le_bytes(bits)
                    })
                    .max()
            }
        }
    }

    fn encoded_size(&self) -> Option<u32> {
        if self.ty.is_scalable() {
            return None;
        }
        let bits = self.ty().element_bits()?;
        bits.div_ceil(8).checked_mul(self.ty.lane_count() as u32)
    }
    fn is_dense(&self) -> bool {
        matches!(&self.data, ConstData::Dense(_))
    }
}

impl From<VectorConst> for Constant {
    fn from(value: VectorConst) -> Self {
        Self::Vector(value)
    }
}
impl TryFrom<Constant> for VectorConst {
    type Error = &'static str;
    fn try_from(value: Constant) -> Result<Self, Self::Error> {
        match value {
            Constant::Vector(value) => Ok(value),
            Constant::Scalar(_) => Err("expected a vector constant"),
        }
    }
}
impl From<ScalarConst> for Constant {
    fn from(value: ScalarConst) -> Self {
        Self::Scalar(value)
    }
}
impl From<Int> for Constant {
    fn from(value: Int) -> Self {
        ScalarConst::from(value).into()
    }
}
impl From<Float> for Constant {
    fn from(value: Float) -> Self {
        ScalarConst::from(value).into()
    }
}
