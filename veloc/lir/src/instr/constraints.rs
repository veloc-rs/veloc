//! Placement requirements apply to operand occurrences, never to an SSA value's
//! entire lifetime. Value categories are independent of placement: both ordinary
//! and state operands can require a fixed physical register.
//! Inputs are read before results are written.
use crate::{OperandRef, PReg, Reg};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Registers(&'static [Reg]),
    Fixed(Reg),
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

/// A physical hardware-state operand declared by an instruction's signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateOperand {
    pub operand: crate::OperandRef,
    pub unit: PReg,
}
