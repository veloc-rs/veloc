//! Veloc MIR type system.

use alloc::vec::Vec;
use core::fmt::{self, Display};
use cranelift_entity::{EntityList, ListPool, entity_impl};

/// Value 列表的内存池
pub type ValueListPool = ListPool<Value>;
/// Value 列表（使用 cranelift-entity 的紧凑表示）
pub type ValueList = EntityList<Value>;

/// Scalar lane types supported by the MIR.
///
/// Signedness is carried by operations rather than types. Discriminant zero is
/// deliberately unused so that [`Type::INVALID`] cannot be mistaken for a real
/// scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ScalarType {
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    F32 = 5,
    F64 = 6,
    Bool = 7,
    /// Opaque pointer. Its size is supplied by the target data layout.
    Ptr = 8,
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::Bool => "bool",
            ScalarType::Ptr => "ptr",
        };
        write!(f, "{}", name)
    }
}

impl ScalarType {
    /// Decode the compact scalar code used by [`Type`] and bytecode metadata.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::I8),
            2 => Some(Self::I16),
            3 => Some(Self::I32),
            4 => Some(Self::I64),
            5 => Some(Self::F32),
            6 => Some(Self::F64),
            7 => Some(Self::Bool),
            8 => Some(Self::Ptr),
            _ => None,
        }
    }

    /// Number of logical bits in one lane, independent of its storage.
    pub const fn bits(self) -> Option<u32> {
        match self {
            Self::Bool => Some(1),
            Self::I8 => Some(8),
            Self::I16 => Some(16),
            Self::I32 | Self::F32 => Some(32),
            Self::I64 | Self::F64 => Some(64),
            Self::Ptr => None,
        }
    }

    /// Size of a non-pointer lane in the MIR's byte representation.
    /// A boolean occupies one byte even though it has one logical bit.
    pub const fn fixed_size_bytes(self) -> Option<u32> {
        match self {
            Self::I8 | Self::Bool => Some(1),
            Self::I16 => Some(2),
            Self::I32 | Self::F32 => Some(4),
            Self::I64 | Self::F64 => Some(8),
            Self::Ptr => None,
        }
    }

    /// 是否是整数类型
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        )
    }

    /// 是否是浮点类型
    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
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

/// 类型表示（位压缩）
///
/// 位域布局 (16 bits):
/// ```text
/// [0..4]   Scalar Type ID (4 bits; zero is invalid)
/// [4..8]   Lane Count Log2 (4 bits): 0=scalar, 1=2lanes, 2=4lanes, ...
/// [8]      Scalable Flag (1 bit): 0=Fixed, 1=Scalable
/// [9..16]  Reserved (7 bits)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Type(u16);

// 位掩码常量
const SCALAR_MASK: u16 = 0x000F; // [0..4]
const LANES_LOG2_MASK: u16 = 0x00F0; // [4..8]
const SCALABLE_MASK: u16 = 0x0100; // [8]
const USED_MASK: u16 = SCALAR_MASK | LANES_LOG2_MASK | SCALABLE_MASK;
const LANES_LOG2_SHIFT: u16 = 4;
const SCALABLE_SHIFT: u16 = 8;

impl Type {
    // === 构造函数 ===

    /// 创建标量类型
    const fn new_scalar(scalar: ScalarType) -> Self {
        Self(scalar as u16)
    }

    /// Create a vector type. The compact MIR encoding supports power-of-two
    /// lane counts from 2 through 32768.
    pub const fn new_vector(element: ScalarType, lanes: u16, scalable: bool) -> Option<Self> {
        if lanes < 2 || !lanes.is_power_of_two() || element as u8 == ScalarType::Ptr as u8 {
            return None;
        }
        let elem_bits = element as u16;
        let log2_lanes = lanes.trailing_zeros() as u16;
        let scalable_bit = if scalable { 1 << SCALABLE_SHIFT } else { 0 };

        Some(Self(
            elem_bits | (log2_lanes << LANES_LOG2_SHIFT) | scalable_bit,
        ))
    }

    /// Create a vector mask. Masks are boolean vectors rather than a separate
    /// type family.
    pub const fn new_mask(lanes: u16, scalable: bool) -> Option<Self> {
        Self::new_vector(ScalarType::Bool, lanes, scalable)
    }

    // === 预定义常量 ===

    pub const INVALID: Self = Self(0);
    pub const I8: Self = Self::new_scalar(ScalarType::I8);
    pub const I16: Self = Self::new_scalar(ScalarType::I16);
    pub const I32: Self = Self::new_scalar(ScalarType::I32);
    pub const I64: Self = Self::new_scalar(ScalarType::I64);
    pub const F32: Self = Self::new_scalar(ScalarType::F32);
    pub const F64: Self = Self::new_scalar(ScalarType::F64);
    pub const BOOL: Self = Self::new_scalar(ScalarType::Bool);
    pub const PTR: Self = Self::new_scalar(ScalarType::Ptr);

    pub const I32X4: Self = Self::vector_unchecked(ScalarType::I32, 4, false);
    pub const I64X2: Self = Self::vector_unchecked(ScalarType::I64, 2, false);
    pub const F32X4: Self = Self::vector_unchecked(ScalarType::F32, 4, false);
    pub const F64X2: Self = Self::vector_unchecked(ScalarType::F64, 2, false);
    pub const I8X16: Self = Self::vector_unchecked(ScalarType::I8, 16, false);
    pub const I16X8: Self = Self::vector_unchecked(ScalarType::I16, 8, false);

    const fn vector_unchecked(element: ScalarType, lanes: u16, scalable: bool) -> Self {
        let scalable_bit = if scalable { 1 << SCALABLE_SHIFT } else { 0 };
        Self(element as u16 | ((lanes.trailing_zeros() as u16) << LANES_LOG2_SHIFT) | scalable_bit)
    }

    // === 访问器 ===

    /// 获取标量类型 ID
    pub fn scalar_type(self) -> ScalarType {
        ScalarType::from_code((self.0 & SCALAR_MASK) as u8)
            .expect("invalid MIR type has no scalar lane type")
    }

    /// 获取通道数的 log2 值
    fn lanes_log2(&self) -> u16 {
        (self.0 & LANES_LOG2_MASK) >> LANES_LOG2_SHIFT
    }

    /// 获取通道数
    pub fn lane_count(&self) -> u16 {
        1 << self.lanes_log2()
    }

    /// 是否是标量类型
    pub fn is_valid(self) -> bool {
        Self::try_from_raw(self.0).is_some()
    }

    pub fn is_scalar(self) -> bool {
        self.0 != 0 && self.lanes_log2() == 0 && !self.is_scalable()
    }

    /// 是否是向量类型（数据向量）
    pub fn is_vector(self) -> bool {
        self.lanes_log2() > 0
    }

    /// 是否是谓词/掩码类型
    pub fn is_predicate(self) -> bool {
        self.is_vector() && self.scalar_type() == ScalarType::Bool
    }

    /// 是否是可伸缩向量
    pub fn is_scalable(self) -> bool {
        (self.0 & SCALABLE_MASK) != 0
    }

    /// 是否是固定长度向量
    pub fn is_fixed(self) -> bool {
        self.is_vector() && !self.is_scalable()
    }

    /// 获取元素类型（向量）或自身（标量）
    pub fn element_type(self) -> Type {
        if self.is_vector() {
            Self(self.0 & SCALAR_MASK)
        } else {
            self
        }
    }

    /// 获取向量形状（通道数 + 是否可伸缩）
    /// 返回 (lanes, scalable)
    pub fn vector_shape(self) -> (u16, bool) {
        (self.lane_count(), self.is_scalable())
    }

    /// 是否是整数类型
    pub fn is_integer(self) -> bool {
        self.is_valid() && self.scalar_type().is_integer()
    }

    /// 是否是指针类型
    pub fn is_ptr(self) -> bool {
        self == Self::PTR
    }

    /// 是否是浮点类型
    pub fn is_float(self) -> bool {
        self.is_valid() && self.scalar_type().is_float()
    }

    /// Logical bits in each scalar lane; pointers require a target layout.
    pub fn element_bits(self) -> Option<u32> {
        self.scalar_type().bits()
    }

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
        let Some(lane_bytes) = self.scalar_type().fixed_size_bytes() else {
            return TypeSize::TargetDependent;
        };
        let min_bytes = lane_bytes * u32::from(self.lane_count());
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

    /// Decode a raw value after validating all currently defined fields.
    pub(crate) const fn try_from_raw(raw: u16) -> Option<Self> {
        if raw == 0 || raw & !USED_MASK != 0 {
            return None;
        }
        let Some(scalar) = ScalarType::from_code((raw & SCALAR_MASK) as u8) else {
            return None;
        };
        let lanes_log2 = (raw & LANES_LOG2_MASK) >> LANES_LOG2_SHIFT;
        let scalable = raw & SCALABLE_MASK != 0;
        if (lanes_log2 == 0 && scalable)
            || (lanes_log2 > 0 && scalar as u8 == ScalarType::Ptr as u8)
        {
            return None;
        }
        Some(Self(raw))
    }
}

impl Default for Type {
    fn default() -> Self {
        Self::INVALID
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
            let elem = self.scalar_type();
            let lanes = self.lane_count();
            if self.is_scalable() {
                write!(f, "<vscale x {} x {:?}>", lanes, elem)
            } else {
                write!(f, "<{} x {:?}>", lanes, elem)
            }
        } else {
            write!(f, "{:?}", self.scalar_type())
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
            let elem = self.scalar_type();
            let lanes = self.lane_count();
            if self.is_scalable() {
                write!(f, "{}<scalable {}>", elem, lanes)
            } else {
                write!(f, "{}<{}>", elem, lanes)
            }
        } else {
            write!(f, "{}", self.scalar_type())
        }
    }
}

// =============================================================================
// Entity Types (using cranelift-entity)
// =============================================================================

/// A reference to a Value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Value(pub u32);
entity_impl!(Value, "v");

/// Data about a value: its type and definition.
#[derive(Debug, Clone)]
pub struct ValueData {
    pub ty: Type,
    pub def: ValueDef,
}

/// Definition of a value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueDef {
    /// Value is defined by an instruction.
    Inst(crate::Inst),
    /// Value is a block parameter.
    Param(Block),
}

/// A reference to a basic block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(pub u32);
entity_impl!(Block, "block");

/// A reference to a block call (branch destination with arguments).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockCall(pub u32);
entity_impl!(BlockCall, "bc");

/// Data for a block call: the target block and arguments.
#[derive(Debug, Clone, Copy)]
pub struct BlockCallData {
    pub block: Block,
    pub args: ValueList,
}

/// A reference to a jump table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JumpTable(pub u32);
entity_impl!(JumpTable, "jt");

/// Data for a jump table: list of block calls.
#[derive(Debug, Clone)]
pub struct JumpTableData {
    pub targets: alloc::vec::Vec<BlockCall>,
}

/// A reference to a stack slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackSlot(pub u32);
entity_impl!(StackSlot, "ss");

/// A reference to a module identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleId(pub u32);
entity_impl!(ModuleId, "module");

/// A reference to a function identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncId(pub u32);
entity_impl!(FuncId, "func");

/// A reference to a signature identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SigId(pub u32);
entity_impl!(SigId, "sig");

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

/// A reference to a variable (SSA variable used in function building).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable(pub u32);
entity_impl!(Variable, "var");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_types() {
        assert!(Type::I32.is_scalar());
        assert!(!Type::I32.is_vector());
        assert_eq!(Type::I32.scalar_type(), ScalarType::I32);
        assert_eq!(Type::I32.storage_size(), TypeSize::Fixed(4));
        assert!(Type::I32.is_integer());
        assert!(!Type::I32.is_float());
    }

    #[test]
    fn test_vector_types() {
        let v4i32 = Type::new_vector(ScalarType::I32, 4, false).unwrap();
        assert!(!v4i32.is_scalar());
        assert!(v4i32.is_vector());
        assert!(!v4i32.is_scalable());
        assert_eq!(v4i32.lane_count(), 4);
        assert_eq!(v4i32.element_type(), Type::I32);
        assert_eq!(v4i32.storage_size(), TypeSize::Fixed(16));
    }

    #[test]
    fn test_scalable_vector() {
        let scalable = Type::new_vector(ScalarType::F32, 4, true).unwrap();
        assert!(scalable.is_vector());
        assert!(scalable.is_scalable());
        assert_eq!(scalable.lane_count(), 4);
        assert_eq!(
            scalable.storage_size(),
            TypeSize::Scalable { min_bytes: 16 }
        );
    }

    #[test]
    fn test_predicate() {
        let mask = Type::new_mask(8, false).unwrap();
        assert!(mask.is_predicate());
        assert!(mask.is_vector());
        assert!(!mask.is_scalar());
        assert_eq!(mask.lane_count(), 8);
    }

    #[test]
    fn test_type_copy() {
        let ty = Type::I64;
        let ty2 = ty; // Copy，不是 Move
        assert_eq!(ty, ty2);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Type::I32), "i32");
        assert_eq!(
            format!("{}", Type::new_vector(ScalarType::I32, 4, false).unwrap()),
            "i32<4>"
        );
        assert_eq!(
            format!("{}", Type::new_vector(ScalarType::F64, 2, true).unwrap()),
            "f64<scalable 2>"
        );
    }

    #[test]
    fn rejects_invalid_encodings() {
        assert_eq!(core::mem::size_of::<Type>(), 2);
        assert_eq!(Type::try_from_raw(0), None);
        assert_eq!(Type::try_from_raw(0xffff), None);
        assert_eq!(Type::new_vector(ScalarType::Ptr, 4, false), None);
        assert_eq!(Type::new_vector(ScalarType::I32, 3, false), None);
    }

    #[test]
    fn test_predicate_type() {
        let mask_fixed = Type::new_mask(8, false).unwrap();
        assert!(mask_fixed.is_predicate());
        assert!(mask_fixed.is_vector());
        assert!(!mask_fixed.is_scalable());
        assert_eq!(mask_fixed.lane_count(), 8);

        let mask_scalable = Type::new_mask(4, true).unwrap();
        assert!(mask_scalable.is_predicate());
        assert!(mask_scalable.is_scalable());
        assert_eq!(mask_scalable.lane_count(), 4);
    }
}
