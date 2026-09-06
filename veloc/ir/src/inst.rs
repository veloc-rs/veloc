use super::dfg::DataFlowGraph;
use crate::opspec::{MemoryEffect, OpFormat};
use crate::types::{BlockCall, FuncId, JumpTable, StackSlot, Value, ValueList};
use crate::{FloatCC, IntCC, Intrinsic, MemFlags, Opcode, SigId};
use core::fmt;
use cranelift_entity::entity_impl;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrIndexImm {
    pub offset: i32,
    pub scale: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PtrIndexImmId(pub u32);
entity_impl!(PtrIndexImmId, "ptr_index_imm");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

// =============================================================================
// Vector Extension IDs (用于指向辅助数据池)
// =============================================================================

/// 向量操作扩展信息 ID (指向 DFG.vector_ext_pool)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorExtId(pub u32);
entity_impl!(VectorExtId, "vext");

/// 常量池 ID (用于 Shuffle 掩码等)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantPoolId(pub u32);
entity_impl!(ConstantPoolId, "const");

/// 常量池中的数据
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantPoolData {
    /// 原始字节数据 (用于向量常量、掩码等)
    Bytes(alloc::vec::Vec<u8>),
}

// =============================================================================
// 向量操作辅助数据结构 (存储在 DFG 的 Arena 中)
// =============================================================================

/// 向量操作扩展信息
/// 用于存储带 Mask 和 EVL 的向量操作（RISC-V V / AVX-512）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorExtData {
    /// 谓词/掩码 (boolean vector)
    pub mask: Value,
    /// 显式向量长度 (Type::I32), None 表示使用默认 VL
    pub evl: Option<Value>,
}

/// Optional configuration shared by vector memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorMemOptions {
    /// 立即数偏移
    pub offset: i32,
    /// 内存标志 (对齐、Volatile等)
    pub flags: MemFlags,
    /// 索引缩放因子 (用于 Gather/Scatter，如 index * scale)
    pub scale: u8,
    /// 掩码 (可选)
    pub mask: Option<Value>,
    /// 显式向量长度 (可选)
    pub evl: Option<Value>,
}

impl Default for VectorMemOptions {
    fn default() -> Self {
        Self {
            offset: 0,
            flags: MemFlags::new(),
            scale: 1,
            mask: None,
            evl: None,
        }
    }
}

/// 扩展配置 ID
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorMemExtId(pub u32);
entity_impl!(VectorMemExtId, "vmem_ext");

impl Inst {
    pub fn visit_operands<F>(self, dfg: &DataFlowGraph, f: F)
    where
        F: FnMut(Value),
    {
        dfg.instructions[self].visit_operands(dfg, f)
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

impl InstructionData {
    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    pub fn memory_effect(&self, dfg: &DataFlowGraph) -> MemoryEffect {
        let effect = self.opcode().spec().memory_effect;
        let flags = self.memory_flags(dfg);
        if flags.is_some_and(|flags| flags.is_volatile()) {
            effect.with_volatile()
        } else {
            effect
        }
    }

    pub fn has_side_effects(&self, dfg: &DataFlowGraph) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator() || spec.may_trap() || self.memory_effect(dfg).has_side_effects()
    }
}

impl fmt::Display for InstructionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}
