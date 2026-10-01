//! Placement requirements apply to operand occurrences, never to an SSA value's
//! entire lifetime, except State values which denote a fixed hardware resource.
//! Inputs are read before results are written.
use crate::{OperandRef, Reg};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Registers(&'static [Reg]),
    Fixed(Reg),
    /// A value resident in a non-renamable hardware state unit. It cannot be
    /// copied or spilled by the ordinary register allocator.
    State(Reg),
    /// A result must reuse this input's location; its SSA identity stays distinct.
    Reuse(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandConstraint {
    pub operand: OperandRef,
    pub placement: Placement,
}

impl OperandConstraint {
    pub const fn fixed(operand: OperandRef, reg: Reg) -> Self {
        Self {
            operand,
            placement: Placement::Fixed(reg),
        }
    }
}
