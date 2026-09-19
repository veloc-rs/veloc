//! SSA edge arguments become physical parallel copies only after allocation.
use super::linear_scan::RegisterAllocator;
use crate::target::SpillKind;
use crate::{Error, Result};
use alloc::format;
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_lir::{InstId, MachineFunction, Reg, StackFrame, StackSlot};
use veloc_mir::Type;

/// A physical move sequence for one selected branch, detached until materialization.
pub struct EdgeAllocation {
    pub(crate) branch: InstId,
    pub(crate) instructions: Vec<InstId>,
}

impl EdgeAllocation {
    pub fn branch(&self) -> InstId {
        self.branch
    }
    pub fn instructions(&self) -> &[InstId] {
        &self.instructions
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Location {
    Reg(Reg),
    Stack(StackSlot),
}

impl RegisterAllocator<'_> {
    fn location(&self, reg: Reg) -> Result<Location> {
        if reg.is_preg() {
            return Ok(Location::Reg(reg));
        }
        if let Some(reg) = self.assigned(reg) {
            return Ok(Location::Reg(reg.into()));
        }
        self.spill_slot(reg)
            .map(Location::Stack)
            .ok_or_else(|| Error::codegen("unallocated edge value"))
    }

    pub(super) fn plan_edges(
        &self,
        f: &mut MachineFunction,
        frame: &mut StackFrame,
    ) -> Result<Vec<EdgeAllocation>> {
        let mut edges = Vec::new();
        let mut cycle_slots = alloc::collections::BTreeMap::new();
        let mut block = f.blocks().next();
        while let Some(current_block) = block {
            let next_block = f.layout().next_block(current_block);
            let mut cursor = f.layout().first_inst(current_block);
            while let Some(id) = cursor {
                let next_id = f.layout().next_inst(id);
                let successor = {
                    let mut successors = f.successors(id);
                    let first = successors.next().map(|edge| {
                        let target = edge.block;
                        let args: SmallVec<[_; 2]> = edge.args.iter().copied().collect();
                        (target, args)
                    });
                    if first.is_some() && successors.next().is_some() {
                        return Err(Error::codegen(
                            "selected branches must carry one explicit edge each",
                        ));
                    }
                    first
                };
                let Some((target_block, args)) = successor else {
                    cursor = next_id;
                    continue;
                };
                let target = &target_block;
                let params = f
                    .block_params(*target)
                    .ok_or_else(|| Error::codegen("unknown edge target"))?;
                if params.len() != args.len() {
                    return Err(Error::codegen("edge argument count mismatch"));
                }
                let mut pending = Vec::new();
                for (&dst, &src) in params.iter().zip(&args) {
                    let ty = f.vreg_data(dst).ty;
                    if ty != f.vreg_data(src).ty {
                        return Err(Error::codegen("edge argument type mismatch"));
                    }
                    let dst = self.location(dst)?;
                    let src = self.location(src)?;
                    if dst != src {
                        pending.push((dst, src, ty));
                    }
                }
                let mut instructions = Vec::new();
                // A destination may be overwritten only after its old value is no
                // longer needed by another move. Save one source to break a cycle.
                while !pending.is_empty() {
                    if let Some(index) = pending
                        .iter()
                        .position(|(dst, _, _)| !pending.iter().any(|(_, src, _)| src == dst))
                    {
                        let (dst, src, ty) = pending.remove(index);
                        self.move_location(f, frame, &mut instructions, dst, src, ty)?;
                    } else {
                        let (_, src, ty) = pending[0];
                        let layout = &self.target.desc().data_layout;
                        let layout = layout.layout_of(ty).ok_or_else(|| {
                            Error::codegen(format!("unknown storage layout: {ty:?}"))
                        })?;
                        let size = layout.alloc_size().ok_or_else(|| {
                            Error::codegen(format!("stack allocation requires fixed size: {ty:?}"))
                        })?;
                        let align = layout.align;
                        let slot = *cycle_slots
                            .entry((size, align))
                            .or_insert_with(|| frame.alloc_slot(size, align));
                        let saved = Location::Stack(slot);
                        self.move_location(f, frame, &mut instructions, saved, src, ty)?;
                        for (_, input, _) in &mut pending {
                            if *input == src {
                                *input = saved;
                            }
                        }
                    }
                }
                if !instructions.is_empty() {
                    instructions.push(self.target.jump_instruction(f.editor().writer(), *target)?);
                    edges.push(EdgeAllocation {
                        branch: id,
                        instructions,
                    });
                }
                cursor = next_id;
            }
            block = next_block;
        }
        Ok(edges)
    }

    fn move_location(
        &self,
        f: &mut MachineFunction,
        frame: &StackFrame,
        out: &mut Vec<InstId>,
        dst: Location,
        src: Location,
        ty: Type,
    ) -> Result<()> {
        match (dst, src) {
            (Location::Reg(dst), Location::Reg(src)) => {
                out.push(
                    self.target
                        .copy_instruction(f.editor().writer(), dst, src, ty)?,
                );
            }
            (Location::Stack(dst), Location::Stack(src)) => {
                let class = self.target.desc().reg_class_for_vreg(&ty, None);
                let scratch =
                    *self.target.spill_scratch(class).first().ok_or_else(|| {
                        Error::codegen("edge stack copy needs a scratch register")
                    })?;
                self.move_location(
                    f,
                    frame,
                    out,
                    Location::Reg(scratch),
                    Location::Stack(src),
                    ty,
                )?;
                self.move_location(
                    f,
                    frame,
                    out,
                    Location::Stack(dst),
                    Location::Reg(scratch),
                    ty,
                )?;
            }
            (Location::Reg(reg), Location::Stack(slot))
            | (Location::Stack(slot), Location::Reg(reg)) => {
                let load = matches!(dst, Location::Reg(_));
                let slot = &frame.slots[slot];
                let fp = self
                    .target
                    .desc()
                    .registers
                    .special_regs
                    .frame_pointer
                    .ok_or_else(|| Error::codegen("edge stack copies require a frame pointer"))?;
                out.push(self.target.spill_instruction(
                    f.editor().writer(),
                    if load {
                        SpillKind::Load
                    } else {
                        SpillKind::Store
                    },
                    reg,
                    slot.base.resolve(fp),
                    slot.offset as i64,
                    ty,
                )?);
            }
        }
        Ok(())
    }
}
