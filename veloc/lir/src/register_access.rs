//! Borrowed register access for analyses that count registers, not operands.
use crate::{InstRef, Reg};
use smallvec::SmallVec;

/// A view of an instruction's explicit and implicit register access.
/// Each iterator yields distinct register identities in first-occurrence order.
/// Physical register aliases must already use the target's storage roots.
#[derive(Clone, Copy)]
pub struct RegisterAccess<'a> {
    inst: InstRef<'a>,
}

impl<'a> RegisterAccess<'a> {
    pub(crate) fn new(inst: InstRef<'a>) -> Self {
        Self { inst }
    }

    /// Explicit inputs, successor arguments and implicit reads.
    pub fn reads(self) -> impl Iterator<Item = Reg> + 'a {
        distinct(self.inst.uses())
    }

    /// Explicit results, implicit writes and ABI clobbers.
    /// Clobbers destroy a register's contents without defining an SSA value.
    pub fn writes(self) -> impl Iterator<Item = Reg> + 'a {
        distinct(self.inst.defs().chain(self.inst.clobbers()))
    }

    /// Registers touched in either direction, counting read/write overlaps once.
    pub fn all(self) -> impl Iterator<Item = Reg> + 'a {
        distinct(
            self.inst
                .uses()
                .chain(self.inst.defs())
                .chain(self.inst.clobbers()),
        )
    }

    pub fn is_read(self, reg: Reg) -> bool {
        self.inst.uses().any(|input| input == reg)
    }
}

fn distinct(regs: impl Iterator<Item = Reg>) -> impl Iterator<Item = Reg> {
    // Ordinary instructions need only inline scratch space. The view itself
    // retains no operand copy or cache beyond this iteration.
    let mut seen = SmallVec::<[Reg; 8]>::new();
    regs.filter(move |reg| {
        if seen.contains(reg) {
            false
        } else {
            seen.push(*reg);
            true
        }
    })
}
