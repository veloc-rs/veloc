//! Resolve simultaneous location transfers at instruction and CFG boundaries.
use super::Transfer;
use crate::target::{SpillKind, TargetRegalloc};
use crate::{Error, Result};
use std::collections::BTreeMap;
use veloc_lir::{Reg, StackBatch, StackSlot, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Location {
    Reg(Reg),
    Stack(StackSlot),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Move {
    pub dst: Location,
    pub src: Location,
    pub ty: Type,
}

#[derive(Default)]
pub(super) struct MoveResolver {
    cycle_slots: BTreeMap<(u32, u32), StackSlot>,
}

impl MoveResolver {
    pub fn resolve(
        &mut self,
        target: &dyn TargetRegalloc,
        frame: &mut StackBatch,
        mut pending: Vec<Move>,
        protected: &[(Reg, Option<Type>)],
    ) -> Result<Vec<Transfer>> {
        pending.retain(|m| m.dst != m.src);
        let mut unique = Vec::<Move>::new();
        for m in pending {
            if let Some(old) = unique.iter().find(|old| old.dst == m.dst) {
                if old.src != m.src || old.ty != m.ty {
                    return Err(Error::codegen("conflicting parallel move destinations"));
                }
            } else {
                unique.push(m);
            }
        }
        let mut pending = unique;
        // Keep scratch helpers from destroying a completed destination or an
        // input still needed by another transfer in this batch.
        let mut protected = protected.to_vec();
        for m in &pending {
            for loc in [m.src, m.dst] {
                if let Location::Reg(reg) = loc {
                    protected.push((reg, Some(m.ty)));
                }
            }
        }
        let mut out = Vec::new();
        while !pending.is_empty() {
            if let Some(index) = pending
                .iter()
                .position(|m| !pending.iter().any(|n| n.src == m.dst))
            {
                let m = pending.remove(index);
                self.emit(target, frame, &mut out, m, &protected)?;
            } else {
                let src = pending[0].src;
                // A physical register can have several width-specific uses.
                // Save the widest one before redirecting all readers.
                let ty = pending
                    .iter()
                    .filter(|m| m.src == src)
                    .map(|m| Ok((storage(target, m.ty)?.0, m.ty)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .max_by_key(|(size, _)| *size)
                    .unwrap()
                    .1;
                let (size, align) = storage(target, ty)?;
                let slot = *self.cycle_slots.entry((size, align)).or_insert_with(|| {
                    frame.alloc_object(veloc_lir::StackObject::Local, size, align)
                });
                let saved = Location::Stack(slot);
                self.emit(
                    target,
                    frame,
                    &mut out,
                    Move {
                        dst: saved,
                        src,
                        ty,
                    },
                    &protected,
                )?;
                for m in &mut pending {
                    if m.src == src {
                        m.src = saved;
                    }
                }
            }
        }
        Ok(out)
    }

    fn emit(
        &self,
        target: &dyn TargetRegalloc,
        frame: &mut StackBatch,
        out: &mut Vec<Transfer>,
        m: Move,
        protected: &[(Reg, Option<Type>)],
    ) -> Result<()> {
        match (m.dst, m.src) {
            (Location::Reg(dst), Location::Reg(src)) => {
                out.push(Transfer::Copy { dst, src, ty: m.ty })
            }
            (Location::Stack(dst), Location::Stack(src)) => {
                let class = target.desc().reg_class_for_vreg(&m.ty, None);
                let scratches = target.spill_scratch(class);
                let free = scratches
                    .iter()
                    .copied()
                    .find(|reg| !protected.iter().any(|(r, _)| r == reg));
                let occupied = scratches.iter().copied().find_map(|reg| {
                    let types = protected
                        .iter()
                        .filter(|(r, _)| *r == reg)
                        .map(|(_, ty)| *ty)
                        .collect::<Option<Vec<_>>>()?;
                    let ty = types
                        .into_iter()
                        .map(|ty| Ok((storage(target, ty)?.0, ty)))
                        .collect::<Result<Vec<_>>>()
                        .ok()?
                        .into_iter()
                        .max_by_key(|(size, _)| *size)?
                        .1;
                    Some((reg, ty))
                });
                let (reg, save_type) = if let Some(reg) = free {
                    (reg, None)
                } else if let Some((reg, ty)) = occupied {
                    (reg, Some(ty))
                } else {
                    return Err(Error::codegen(
                        "stack transfer needs a scratch with known storage",
                    ));
                };
                // Preserve the actual value held by a borrowed scratch, using
                // its widest live representation rather than guessing from a bank.
                let saved = if let Some(ty) = save_type {
                    let (size, align) = storage(target, ty)?;
                    let slot = frame.alloc_object(veloc_lir::StackObject::Local, size, align);
                    out.push(Transfer::Spill {
                        kind: SpillKind::Store,
                        reg,
                        slot,
                        ty,
                    });
                    Some((slot, ty))
                } else {
                    None
                };
                out.push(Transfer::Spill {
                    kind: SpillKind::Load,
                    reg,
                    slot: src,
                    ty: m.ty,
                });
                out.push(Transfer::Spill {
                    kind: SpillKind::Store,
                    reg,
                    slot: dst,
                    ty: m.ty,
                });
                if let Some((slot, ty)) = saved {
                    out.push(Transfer::Spill {
                        kind: SpillKind::Load,
                        reg,
                        slot,
                        ty,
                    });
                }
            }
            (Location::Reg(reg), Location::Stack(slot)) => out.push(Transfer::Spill {
                kind: SpillKind::Load,
                reg,
                slot,
                ty: m.ty,
            }),
            (Location::Stack(slot), Location::Reg(reg)) => out.push(Transfer::Spill {
                kind: SpillKind::Store,
                reg,
                slot,
                ty: m.ty,
            }),
        }
        Ok(())
    }
}

pub(super) fn storage(target: &dyn TargetRegalloc, ty: Type) -> Result<(u32, u32)> {
    let layout = target
        .desc()
        .data_layout
        .layout_of(ty)
        .ok_or_else(|| Error::codegen("unknown transfer storage layout"))?;
    Ok((
        layout
            .alloc_size()
            .ok_or_else(|| Error::codegen("transfer requires fixed storage size"))?,
        layout.align,
    ))
}
