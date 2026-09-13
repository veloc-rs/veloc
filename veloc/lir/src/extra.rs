//! Low-level IR (LIR) 指令附加信息定义

use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_mir::{Block, Signature};

use crate::Reg;

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

impl InstExtra {
    /// One flattened order for edge-argument traversal and controlled editing.
    pub fn edge_args(&self) -> impl Iterator<Item = Reg> + '_ {
        let (first, second, table): (&[Reg], &[Reg], &[BrTableTarget]) = match self {
            Self::Branch(info) => (&info.args, &[], &[]),
            Self::BranchCond(info) => (&info.then_args, &info.else_args, &[]),
            Self::BrTable(info) => (&[], &[], &info.targets),
            Self::Call(_) | Self::AMode(_) => (&[], &[], &[]),
        };
        first
            .iter()
            .chain(second)
            .chain(table.iter().flat_map(|t| &t.args))
            .copied()
    }

    pub(crate) fn edge_arg_mut(&mut self, mut index: usize) -> &mut Reg {
        match self {
            Self::Branch(info) => &mut info.args[index],
            Self::BranchCond(info) => {
                if index < info.then_args.len() {
                    &mut info.then_args[index]
                } else {
                    &mut info.else_args[index - info.then_args.len()]
                }
            }
            Self::BrTable(info) => {
                for target in &mut info.targets {
                    if index < target.args.len() {
                        return &mut target.args[index];
                    }
                    index -= target.args.len();
                }
                panic!("edge argument index out of bounds");
            }
            Self::Call(_) | Self::AMode(_) => panic!("payload has no edge arguments"),
        }
    }
}
