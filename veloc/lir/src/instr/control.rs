//! Instruction-owned call contracts and logical control-flow edges.
use crate::{BlockId, Reg, StackSlot};
use smallvec::SmallVec;
use veloc_mir::Signature;

impl crate::ControlFlow {
    /// Local branch targets are stored in the instruction's successor operands.
    pub const fn has_explicit_successors(self) -> bool {
        matches!(self, Self::Branch | Self::Jump)
    }

    /// A modeled path continues at the next instruction (or layout fallthrough).
    /// Branch has both a taken path and a continuation; Jump has only targets,
    /// including generic conditional branches with two explicit successors.
    pub const fn may_continue(self) -> bool {
        matches!(self, Self::Next | Self::Call | Self::Branch)
    }

    /// A modeled path leaves the block without executing the next instruction.
    /// Ordinary calls resume locally; their other effects are described separately.
    pub const fn may_leave_block(self) -> bool {
        matches!(self, Self::Branch | Self::Jump | Self::Return | Self::Trap)
    }
}

/// Function-local identity of a control-flow edge. Moving an edge to a selected
/// instruction preserves its ID; copying an instruction creates fresh edges.
/// Deleted IDs are not reused, so optional analysis tables cannot alias new edges.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(u32);
cranelift_entity::entity_impl!(EdgeId, "edge");

/// Outgoing stack space, including ABI-reserved space and argument padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArea {
    pub size: u32,
    pub align: u32,
}

/// A symbolic, function-local outgoing call frame, not an SP displacement.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallFrameId(u32);
cranelift_entity::entity_impl!(CallFrameId, "callframe");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallInfo {
    /// Declared signature. Actual argument types belong to the call operands.
    pub sig: Signature,
    /// Registers whose pre-call contents cannot survive this call.
    pub clobbers: crate::RegMask,
    /// None until ABI lowering. Zero-sized frames still identify lowered calls.
    pub frame: Option<CallFrameId>,
    /// Outgoing stack locations read by this call. Calls remain memory barriers;
    /// these locations refine the boundary rather than replacing other effects.
    pub stack_args: SmallVec<[StackSlot; 2]>,
}

/// An edge occurrence, not just a destination. Two edges may have the same block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Successor<A = SmallVec<[Reg; 2]>> {
    pub block: BlockId,
    pub args: A,
}
