//! Veloc IR Type System

use alloc::vec::Vec;
use core::fmt;
use cranelift_entity::{EntityList, ListPool, entity_impl};

/// Value 列表的内存池
pub type ValueListPool = ListPool<Value>;
/// Value 列表（使用 cranelift-entity 的紧凑表示）
pub type ValueList = EntityList<Value>;

/// 标量类型枚举 (4 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ScalarType {
    // 整数类型 (0-7)
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
    // 预留: I128 = 4

    // 浮点类型 (4-7)
    F32 = 4,
    F64 = 5,
    // 预留: F16 = 6, F128 = 7

    // 特殊类型 (8-15)
    Bool = 8,
    Ptr = 9,   // 指针类型 (不透明，大小取决于目标平台)
    Void = 10, // 无返回值
    EVL = 11,  // 显式向量长度 (Explicit Vector Length)
               // 预留: 12-15
}

impl ScalarType {
    /// 获取标量类型的大小（字节）
    pub fn size_bytes(&self) -> usize {
        match self {
            ScalarType::I8 => 1,
            ScalarType::I16 => 2,
            ScalarType::I32 => 4,
            ScalarType::I64 => 8,
            ScalarType::F32 => 4,
            ScalarType::F64 => 8,
            ScalarType::Bool => 1,
            ScalarType::Ptr => 8, // 假设 64 位平台
            ScalarType::Void => 0,
            ScalarType::EVL => 8, // EVL 是 64 位整数
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

/// 类型表示（位压缩）
///
/// 位域布局 (16 bits):
/// ```text
/// [0..4]   Scalar Type ID (4 bits)
/// [4..8]   Lane Count Log2 (4 bits): 0=scalar, 1=2lanes, 2=4lanes, ...
/// [8]      Scalable Flag (1 bit): 0=Fixed, 1=Scalable
/// [9]      Predicate Flag (1 bit): 0=Data, 1=Predicate/Mask
/// [10..16] Reserved (6 bits)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Type(u16);

// 位掩码常量
const SCALAR_MASK: u16 = 0x000F; // [0..4]
const LANES_LOG2_MASK: u16 = 0x00F0; // [4..8]
const SCALABLE_MASK: u16 = 0x0100; // [8]
const PREDICATE_MASK: u16 = 0x0200; // [9]
const LANES_LOG2_SHIFT: u16 = 4;
const SCALABLE_SHIFT: u16 = 8;
const PREDICATE_SHIFT: u16 = 9;

impl Type {
    // === 构造函数 ===

    /// 创建标量类型
    const fn new_scalar(scalar: ScalarType) -> Self {
        Self(scalar as u16)
    }

    /// 创建向量类型
    pub const fn new_vector(element: ScalarType, lanes: u16, scalable: bool) -> Self {
        debug_assert!(lanes.is_power_of_two(), "Lane count must be power of 2");
        let elem_bits = element as u16;
        let log2_lanes = lanes.trailing_zeros() as u16;
        let scalable_bit = if scalable { 1 << SCALABLE_SHIFT } else { 0 };

        Self(elem_bits | (log2_lanes << LANES_LOG2_SHIFT) | scalable_bit)
    }

    /// 创建谓词/掩码类型
    pub const fn new_predicate(lanes: u16, scalable: bool) -> Self {
        debug_assert!(lanes.is_power_of_two(), "Lane count must be power of 2");
        let log2_lanes = lanes.trailing_zeros() as u16;
        let scalable_bit = if scalable { 1 << SCALABLE_SHIFT } else { 0 };

        Self(
            ScalarType::Bool as u16
                | (log2_lanes << LANES_LOG2_SHIFT)
                | scalable_bit
                | (1 << PREDICATE_SHIFT),
        )
    }

    // === 预定义常量 ===

    pub const I8: Self = Self::new_scalar(ScalarType::I8);
    pub const I16: Self = Self::new_scalar(ScalarType::I16);
    pub const I32: Self = Self::new_scalar(ScalarType::I32);
    pub const I64: Self = Self::new_scalar(ScalarType::I64);
    pub const F32: Self = Self::new_scalar(ScalarType::F32);
    pub const F64: Self = Self::new_scalar(ScalarType::F64);
    pub const BOOL: Self = Self::new_scalar(ScalarType::Bool);
    pub const PTR: Self = Self::new_scalar(ScalarType::Ptr);
    pub const VOID: Self = Self::new_scalar(ScalarType::Void);
    pub const EVL: Self = Self::new_scalar(ScalarType::EVL);

    pub const I32X4: Self = Self::new_vector(ScalarType::I32, 4, false);
    pub const I64X2: Self = Self::new_vector(ScalarType::I64, 2, false);
    pub const F32X4: Self = Self::new_vector(ScalarType::F32, 4, false);
    pub const F64X2: Self = Self::new_vector(ScalarType::F64, 2, false);
    pub const I8X16: Self = Self::new_vector(ScalarType::I8, 16, false);
    pub const I16X8: Self = Self::new_vector(ScalarType::I16, 8, false);

    // === 访问器 ===

    /// 获取标量类型 ID
    pub fn scalar_type(&self) -> ScalarType {
        // 安全：我们只存储有效的标量类型 ID
        unsafe { core::mem::transmute((self.0 & SCALAR_MASK) as u8) }
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
    pub fn is_scalar(&self) -> bool {
        self.lanes_log2() == 0 && !self.is_predicate()
    }

    /// 是否是向量类型（数据向量）
    pub fn is_vector(&self) -> bool {
        self.lanes_log2() > 0 && !self.is_predicate()
    }

    /// 是否是谓词/掩码类型
    pub fn is_predicate(&self) -> bool {
        (self.0 & PREDICATE_MASK) != 0
    }

    /// 是否是可伸缩向量
    pub fn is_scalable(&self) -> bool {
        (self.0 & SCALABLE_MASK) != 0
    }

    /// 是否是固定长度向量
    pub fn is_fixed(&self) -> bool {
        self.is_vector() && !self.is_scalable()
    }

    /// 获取元素类型（向量）或自身（标量）
    pub fn element_type(&self) -> Type {
        if self.is_vector() || self.is_predicate() {
            // 创建相同标量类型的标量版本
            Self(self.0 & SCALAR_MASK)
        } else {
            *self
        }
    }

    /// 获取向量形状（通道数 + 是否可伸缩）
    /// 返回 (lanes, scalable)
    pub fn vector_shape(&self) -> (u16, bool) {
        (self.lane_count(), self.is_scalable())
    }

    /// 是否是整数类型
    pub fn is_integer(&self) -> bool {
        self.scalar_type().is_integer()
    }

    /// 是否是浮点类型
    pub fn is_float(&self) -> bool {
        self.scalar_type().is_float()
    }

    /// 类型大小（字节）
    /// 对于可伸缩向量，返回 min_lanes * element_size
    pub fn size_bytes(&self) -> usize {
        let scalar_size = self.scalar_type().size_bytes();
        let lanes = self.lane_count() as usize;
        scalar_size * lanes
    }

    /// 类型的位宽
    pub fn bit_width(&self) -> usize {
        self.size_bytes() * 8
    }

    /// 获取原始内部值（用于调试/序列化）
    pub fn raw(&self) -> u16 {
        self.0
    }

    /// 从原始值创建（用于反序列化）
    pub fn from_raw(raw: u16) -> Self {
        Self(raw)
    }
}

impl Default for Type {
    fn default() -> Self {
        Self::VOID
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                write!(f, "{}<scalable {}>", elem_name(&elem), lanes)
            } else {
                write!(f, "{}<{}>", elem_name(&elem), lanes)
            }
        } else {
            write!(f, "{}", elem_name(&self.scalar_type()))
        }
    }
}

fn elem_name(scalar: &ScalarType) -> &'static str {
    match scalar {
        ScalarType::I8 => "i8",
        ScalarType::I16 => "i16",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::Bool => "bool",
        ScalarType::Ptr => "ptr",
        ScalarType::Void => "void",
        ScalarType::EVL => "evl",
    }
}

/// 向量种类（兼容旧 API）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorKind {
    Fixed(u16),
    Scalable { min_lanes: u16 },
}

impl VectorKind {
    /// 获取最小通道数
    pub fn min_lanes(&self) -> u16 {
        match self {
            VectorKind::Fixed(lanes) => *lanes,
            VectorKind::Scalable { min_lanes } => *min_lanes,
        }
    }

    /// 是否是可伸缩向量
    pub fn is_scalable(&self) -> bool {
        matches!(self, VectorKind::Scalable { .. })
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
        assert_eq!(Type::I32.size_bytes(), 4);
        assert!(Type::I32.is_integer());
        assert!(!Type::I32.is_float());
    }

    #[test]
    fn test_vector_types() {
        let v4i32 = Type::new_vector(ScalarType::I32, 4, false);
        assert!(!v4i32.is_scalar());
        assert!(v4i32.is_vector());
        assert!(!v4i32.is_scalable());
        assert_eq!(v4i32.lane_count(), 4);
        assert_eq!(v4i32.element_type(), Type::I32);
        assert_eq!(v4i32.size_bytes(), 16);
    }

    #[test]
    fn test_scalable_vector() {
        let scalable = Type::new_vector(ScalarType::F32, 4, true);
        assert!(scalable.is_vector());
        assert!(scalable.is_scalable());
        assert_eq!(scalable.lane_count(), 4);
    }

    #[test]
    fn test_predicate() {
        let mask = Type::new_predicate(8, false);
        assert!(mask.is_predicate());
        assert!(!mask.is_vector());
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
            format!("{}", Type::new_vector(ScalarType::I32, 4, false)),
            "i32<4>"
        );
        assert_eq!(
            format!("{}", Type::new_vector(ScalarType::F64, 2, true)),
            "f64<scalable 2>"
        );
    }

    #[test]
    fn test_evl_type() {
        assert_eq!(Type::EVL.scalar_type(), ScalarType::EVL);
        assert!(Type::EVL.is_scalar());
        assert!(!Type::EVL.is_vector());
        assert!(!Type::EVL.is_predicate());
    }

    #[test]
    fn test_predicate_type() {
        let mask_fixed = Type::new_predicate(8, false);
        assert!(mask_fixed.is_predicate());
        assert!(!mask_fixed.is_vector());
        assert!(!mask_fixed.is_scalable());
        assert_eq!(mask_fixed.lane_count(), 8);

        let mask_scalable = Type::new_predicate(4, true);
        assert!(mask_scalable.is_predicate());
        assert!(mask_scalable.is_scalable());
        assert_eq!(mask_scalable.lane_count(), 4);
    }

    #[test]
    fn test_vector_kind() {
        let fixed = VectorKind::Fixed(8);
        assert_eq!(fixed.min_lanes(), 8);
        assert!(!fixed.is_scalable());

        let scalable = VectorKind::Scalable { min_lanes: 4 };
        assert_eq!(scalable.min_lanes(), 4);
        assert!(scalable.is_scalable());
    }
}
