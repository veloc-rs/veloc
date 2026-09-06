//! Low-level IR (LIR) 指令附加信息定义

use alloc::vec::Vec;
use cranelift_entity::entity_impl;
use smallvec::SmallVec;
use veloc_mir::{Block, Signature};

use crate::Reg;

/// 指令额外信息索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstExtraId(u32);
entity_impl!(InstExtraId, "inst_extra");

/// 调用指令的附加信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallInfo {
    pub sig: Signature,
}

/// `br_table` 的单个目标。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrTableTarget {
    pub block: Block,
    pub args: SmallVec<[Reg; 2]>,
}

/// 无条件分支的边参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchInfo {
    pub args: SmallVec<[Reg; 2]>,
}

/// 条件分支的边参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchCondInfo {
    pub then_args: SmallVec<[Reg; 2]>,
    pub else_args: SmallVec<[Reg; 2]>,
}

/// 跳转表附加信息。
///
/// `targets` 的最后一个元素是 default 目标。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrTableInfo {
    pub targets: Vec<BrTableTarget>,
}

/// 寻址更新模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AMode {
    /// 前索引模式：ptr = ptr + offset, addr = ptr (LLVM Pre-Indexed)
    PreIndex,
    /// 后索引模式：addr = ptr, ptr = ptr + offset (LLVM Post-Indexed)
    PostIndex,
}

/// 少数复杂 LIR 指令的附加 payload。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstExtra {
    Call(CallInfo),
    Branch(BranchInfo),
    BranchCond(BranchCondInfo),
    BrTable(BrTableInfo),
    AMode(AMode),
}
