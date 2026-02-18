use alloc::vec::Vec;
use core::fmt;
use cranelift_entity::{EntityList, ListPool, entity_impl};

use crate::CallConv;
use crate::constant::Constant;
use crate::inst::Inst;

/// Value 列表的内存池
pub type ValueListPool = ListPool<Value>;
/// Value 列表（使用 cranelift-entity 的紧凑表示）
pub type ValueList = EntityList<Value>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(pub u32);
entity_impl!(Value, "v");

impl Value {
    pub fn as_const(self, dfg: &crate::dfg::DataFlowGraph) -> Option<Constant> {
        dfg.as_const(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(pub u32);
entity_impl!(Block, "block");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackSlot(pub u32);
entity_impl!(StackSlot, "ss");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncId(pub u32);
entity_impl!(FuncId, "func");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable(pub u32);
entity_impl!(Variable, "var");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JumpTable(pub u32);
entity_impl!(JumpTable, "jt");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockCall(pub u32);
entity_impl!(BlockCall, "bc");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SigId(pub u32);
entity_impl!(SigId, "sig");

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    pub params: Box<[Type]>,
    /// 支持多返回值（如 WebAssembly 多返回值提案）
    pub ret: Box<[Type]>,
    pub call_conv: CallConv,
}

impl Signature {
    /// 创建函数签名（支持多返回值）
    pub fn new(params: Vec<Type>, ret: Vec<Type>, call_conv: CallConv) -> Self {
        Self {
            params: params.into_boxed_slice(),
            ret: ret.into_boxed_slice(),
            call_conv,
        }
    }

    /// 获取返回值数量
    pub fn num_rets(&self) -> usize {
        self.ret.len()
    }

    /// 是否是多返回值
    pub fn is_multi_value(&self) -> bool {
        self.ret.len() > 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueDef {
    Inst(Inst),
    Param(Block),
}

#[derive(Debug, Clone)]
pub struct ValueData {
    pub ty: Type,
    pub def: ValueDef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockCallData {
    pub block: Block,
    pub args: ValueList,
}

#[derive(Debug, Clone)]
pub struct JumpTableData {
    pub targets: Box<[BlockCall]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Bool,
    Ptr,
    Void,
    /// 多值标记，表示这是一个多返回值聚合值
    /// 具体类型信息需要通过上下文（如函数签名）获取
    MultiValue,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Type::I8 => "i8",
            Type::I16 => "i16",
            Type::I32 => "i32",
            Type::I64 => "i64",
            Type::F32 => "f32",
            Type::F64 => "f64",
            Type::Bool => "bool",
            Type::Ptr => "ptr",
            Type::Void => "void",
            Type::MultiValue => "multi",
        };
        write!(f, "{}", s)
    }
}

impl Type {
    pub fn is_integer(self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }

    pub fn is_float(self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    pub fn size_bytes(&self) -> u32 {
        match self {
            Type::I8 | Type::Bool => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 | Type::Ptr => 8,
            Type::Void | Type::MultiValue => 0,
        }
    }
}
