//! Block-local facts about private allocator spill slots. These slots cannot
//! escape to user code, so only planned stores change their contents.
use super::allocation::Transfer;
use crate::target::SpillKind;
use veloc_lir::{InstRef, Reg, StackSlot, Type};

#[derive(Default)]
pub(super) struct SpillCache(Vec<(Reg, StackSlot, Type)>);

impl SpillCache {
    pub fn invalidate(&mut self, inst: InstRef<'_>) {
        self.0
            .retain(|(reg, _, _)| !inst.register_access().writes().any(|write| write == *reg));
    }

    /// Reuse only full-register representations. Narrow spills may normalize
    /// upper bits differently from a copy, even when their low bits agree.
    pub fn simplify(&self, transfer: Transfer) -> Option<Transfer> {
        if let Transfer::Spill {
            kind: SpillKind::Load,
            reg,
            slot,
            ty,
        } = transfer
            && matches!(ty, Type::I64 | Type::PTR | Type::F64)
            && let Some(&(source, _, _)) = self.0.iter().find(|(_, s, t)| *s == slot && *t == ty)
        {
            return (source != reg).then_some(Transfer::Copy {
                dst: reg,
                src: source,
                ty,
            });
        }
        Some(transfer)
    }

    /// Record the original transfer, including a load replaced by a copy.
    pub fn record(&mut self, transfer: Transfer) {
        match transfer {
            Transfer::Spill {
                kind,
                reg,
                slot,
                ty,
            } => {
                if kind == SpillKind::Store {
                    self.0.retain(|(_, s, _)| *s != slot);
                }
                self.0.retain(|(r, _, _)| *r != reg);
                if matches!(ty, Type::I64 | Type::PTR | Type::F64) {
                    self.0.push((reg, slot, ty));
                }
            }
            Transfer::Copy { .. } | Transfer::Rematerialize { .. } => {}
        }
    }
}
