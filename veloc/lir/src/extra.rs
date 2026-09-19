//! Low-level IR (LIR) 指令附加信息定义

use crate::BlockId as Block;
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_mir::Signature;

use crate::Reg;

/// 调用指令的附加信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallInfo {
    pub sig: Signature,
}

/// `br_table` 的单个目标。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrTableTarget<A = SmallVec<[Reg; 2]>> {
    pub block: Block,
    pub args: A,
}

/// 无条件分支的边参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchInfo<A = SmallVec<[Reg; 2]>> {
    pub args: A,
}

/// 条件分支的边参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchCondInfo<A = SmallVec<[Reg; 2]>> {
    pub then_args: A,
    pub else_args: A,
}

/// 跳转表附加信息。
///
/// `targets` 的最后一个元素是 default 目标。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrTableInfo<A = SmallVec<[Reg; 2]>> {
    pub targets: Vec<BrTableTarget<A>>,
}

/// 少数复杂 LIR 指令的附加 payload。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstExtra<A = SmallVec<[Reg; 2]>> {
    Call(CallInfo),
    Branch(BranchInfo<A>),
    BranchCond(BranchCondInfo<A>),
    BrTable(BrTableInfo<A>),
}

impl<A> InstExtra<A> {
    pub(crate) fn map_args<B>(self, mut f: impl FnMut(A) -> B) -> InstExtra<B> {
        match self {
            Self::Call(info) => InstExtra::Call(info),
            Self::Branch(info) => InstExtra::Branch(BranchInfo { args: f(info.args) }),
            Self::BranchCond(info) => InstExtra::BranchCond(BranchCondInfo {
                then_args: f(info.then_args),
                else_args: f(info.else_args),
            }),
            Self::BrTable(info) => InstExtra::BrTable(BrTableInfo {
                targets: info
                    .targets
                    .into_iter()
                    .map(|t| BrTableTarget {
                        block: t.block,
                        args: f(t.args),
                    })
                    .collect(),
            }),
        }
    }
    pub(crate) fn arg_ranges(&self) -> impl Iterator<Item = &A> {
        let (first, second, table): (Option<&A>, Option<&A>, &[BrTableTarget<A>]) = match self {
            Self::Branch(info) => (Some(&info.args), None, &[]),
            Self::BranchCond(info) => (Some(&info.then_args), Some(&info.else_args), &[]),
            Self::BrTable(info) => (None, None, &info.targets),
            _ => (None, None, &[]),
        };
        first
            .into_iter()
            .chain(second)
            .chain(table.iter().map(|t| &t.args))
    }
}

/// Borrowed payload view. Reading edge arguments never copies their registers.
#[derive(Debug)]
pub enum InstExtraRef<'a> {
    Call(&'a CallInfo),
    Branch(BranchInfo<&'a [Reg]>),
    BranchCond(BranchCondInfo<&'a [Reg]>),
    BrTable(BrTableRef<'a>),
}
#[derive(Debug)]
pub struct BrTableRef<'a> {
    pub(crate) store: &'a crate::InstStore,
    pub(crate) info: &'a BrTableInfo<crate::store::Range>,
}
impl<'a> BrTableRef<'a> {
    pub fn targets(
        &self,
    ) -> impl DoubleEndedIterator<Item = BrTableTarget<&'a [Reg]>> + ExactSizeIterator + 'a {
        let store = self.store;
        self.info.targets.iter().map(move |t| BrTableTarget {
            block: t.block,
            args: store.registers(t.args),
        })
    }
}
impl InstExtraRef<'_> {
    pub fn to_owned(&self) -> InstExtra {
        match self {
            Self::Call(info) => InstExtra::Call((*info).clone()),
            Self::Branch(info) => InstExtra::Branch(BranchInfo {
                args: info.args.into(),
            }),
            Self::BranchCond(info) => InstExtra::BranchCond(BranchCondInfo {
                then_args: info.then_args.into(),
                else_args: info.else_args.into(),
            }),
            Self::BrTable(info) => InstExtra::BrTable(BrTableInfo {
                targets: info
                    .targets()
                    .map(|t| BrTableTarget {
                        block: t.block,
                        args: t.args.into(),
                    })
                    .collect(),
            }),
        }
    }
}
