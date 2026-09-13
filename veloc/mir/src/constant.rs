//! Exact constants. Scalar views add guarantees, not another representation tag.
use crate::{InstWriter, ScalarType, Type, VectorType, dfg::DataFlowGraph, inst::ConstantPoolId};
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

impl InstWriter<'_> {
    pub fn scalar_const(self, value: ScalarConst) -> crate::Inst {
        match value.ty {
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 => {
                self.iconst(Int(value))
            }
            ScalarType::F32 | ScalarType::F64 => self.fconst(Float(value)),
            ScalarType::BOOL => self.bconst(value.bits != 0),
            ScalarType::PTR => unreachable!("pointer constants are not represented by ScalarConst"),
        }
    }
}

/// Storage forms, independent of the MIR value's scalar/vector type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstData {
    Bits(u64),
    /// Immutable little-endian lane bytes in the containing function's pool.
    Dense(ConstantPoolId),
    /// Repeat one scalar bit pattern, also valid for scalable vectors.
    Splat(u64),
}

/// General constant descriptor. Dense handles belong to their originating DFG;
/// copy the bytes into the destination pool when moving between functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Constant {
    ty: Type,
    data: ConstData,
}

impl Constant {
    pub const fn ty(self) -> Type {
        self.ty
    }
    pub const fn data(self) -> ConstData {
        self.data
    }
    pub fn as_scalar(self) -> Option<ScalarConst> {
        let ConstData::Bits(bits) = self.data else {
            return None;
        };
        Some(ScalarConst {
            ty: self.ty.as_scalar().expect("scalar constant"),
            bits,
        })
    }
    pub fn as_vector(self) -> Option<VectorConst> {
        self.ty.as_vector().map(|_| VectorConst(self))
    }
}

/// A vector-typed constant descriptor. Construction fixes shape and storage form,
/// but does not scan dense bytes; validate explicitly before consuming untrusted IR.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorConst(Constant);

impl VectorConst {
    pub const fn dense(ty: VectorType, id: ConstantPoolId) -> Self {
        Self(Constant {
            ty: ty.as_type(),
            data: ConstData::Dense(id),
        })
    }

    /// Derive the vector's element type from the scalar, so they cannot disagree.
    pub fn splat(value: ScalarConst, lanes: u16, scalable: bool) -> Option<Self> {
        let ty = value.ty.vector(lanes, scalable)?.as_type();
        Some(Self(Constant {
            ty,
            data: ConstData::Splat(value.bits),
        }))
    }

    pub const fn ty(self) -> Type {
        self.0.ty
    }
    pub const fn data(self) -> ConstData {
        self.0.data
    }

    pub fn bytes(self, dfg: &DataFlowGraph) -> Option<&[u8]> {
        match self.data() {
            ConstData::Dense(id) => id.get(dfg),
            _ => None,
        }
    }

    pub fn splat_value(self) -> Option<ScalarConst> {
        let ConstData::Splat(bits) = self.data() else {
            return None;
        };
        Some(ScalarConst {
            ty: self
                .ty()
                .as_vector()
                .expect("vector constant")
                .element_type(),
            bits,
        })
    }
}

const impl crate::type_methods::VectorConstInfo for VectorConst {
    fn is_dense(self) -> bool {
        matches!(self.data(), ConstData::Dense(_))
    }
}

impl From<VectorConst> for Constant {
    fn from(value: VectorConst) -> Self {
        value.0
    }
}

impl TryFrom<Constant> for VectorConst {
    type Error = &'static str;
    fn try_from(value: Constant) -> Result<Self, Self::Error> {
        value.as_vector().ok_or("expected a vector constant")
    }
}

impl From<ScalarConst> for Constant {
    fn from(value: ScalarConst) -> Self {
        Self {
            ty: value.ty(),
            data: ConstData::Bits(value.to_bits()),
        }
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

/// Byte-aligned literal storage avoids padding around a scalar's type tag.
/// Conversion uses ordinary byte loads, never unaligned references or unsafe.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarBits {
    bits: [u8; 8],
    ty: crate::ScalarType,
}

impl ScalarBits {
    pub fn new(value: crate::ScalarConst) -> Self {
        Self {
            bits: value.to_bits().to_le_bytes(),
            ty: value.ty,
        }
    }

    pub fn int(self) -> crate::Int {
        Int(ScalarConst {
            ty: self.ty,
            bits: u64::from_le_bytes(self.bits),
        })
    }

    pub fn float(self) -> crate::Float {
        Float(ScalarConst {
            ty: self.ty,
            bits: u64::from_le_bytes(self.bits),
        })
    }
}

/// Vectors also fit inline: compact type, storage kind and exact lane bits/ID.
/// Dense data remains in the existing constant pool; no second pool lookup.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VectorBits {
    payload: [u8; 8],
    ty: [u8; 2],
    splat: bool,
}

impl VectorBits {
    pub fn new(value: crate::VectorConst) -> Self {
        let (payload, splat) = match value.data() {
            crate::ConstData::Dense(id) => (u64::from(id.0), false),
            crate::ConstData::Splat(bits) => (bits, true),
            crate::ConstData::Bits(_) => unreachable!("vector storage form"),
        };
        Self {
            payload: payload.to_le_bytes(),
            ty: value.ty().to_raw().to_le_bytes(),
            splat,
        }
    }

    pub fn value(self) -> crate::VectorConst {
        let ty = crate::Type::from_raw(u16::from_le_bytes(self.ty)).expect("stored vector type");
        let bits = u64::from_le_bytes(self.payload);
        let data = if self.splat {
            ConstData::Splat(bits)
        } else {
            ConstData::Dense(ConstantPoolId(bits as u32))
        };
        VectorConst(Constant { ty, data })
    }
}
