//! Calls and logical control-flow edges.
use crate::{BlockId, Reg};
use smallvec::SmallVec;
use veloc_mir::Signature;

/// Function-local identity of a control-flow edge. Moving an edge to a selected
/// instruction preserves its ID; copying an instruction creates fresh edges.
/// Deleted IDs are not reused, so optional analysis tables cannot alias new edges.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(u32);
cranelift_entity::entity_impl!(EdgeId, "edge");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallInfo {
    pub sig: Signature,
}

/// An edge occurrence, not just a destination. Two edges may have the same block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Successor<A = SmallVec<[Reg; 2]>> {
    pub block: BlockId,
    pub args: A,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstExtra {
    Call(CallInfo),
}

#[derive(Debug)]
pub enum InstExtraRef<'a> {
    Call(&'a CallInfo),
}

impl InstExtraRef<'_> {
    pub fn to_owned(&self) -> InstExtra {
        match self {
            Self::Call(info) => InstExtra::Call((*info).clone()),
        }
    }
}
