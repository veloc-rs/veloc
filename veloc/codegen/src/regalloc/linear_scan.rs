//! Global linear scan with CFG liveness, fixed registers and whole-range spills.
use super::allocation::Transfer;
use super::allocation::{Allocation, InstAllocation};
use crate::analysis::FunctionAnalysisCtx;
use crate::target::{RegClass, SpillKind, TargetRegalloc};
use crate::{Error, Result};
use cranelift_entity::SecondaryMap;
use hashbrown::HashMap;
use std::format;
use std::vec::Vec;
use veloc_lir::{InstId, MachineFunction, PReg, Reg, StackBatch, StackSlot, Type, VReg};

/// Physical occupancy after whole-range allocation. Register IDs denote storage
/// roots (e.g. eax/rax share one ID), not independently allocatable views.
#[derive(Default)]
struct RegisterLiveness {
    fixed: Vec<Vec<(u32, u32)>>,
    assigned: Vec<Vec<(u32, u32, Type)>>,
}

impl RegisterLiveness {
    fn fixed_at(&self, reg: Reg, pos: u32) -> bool {
        self.fixed.get(reg.index() as usize).is_some_and(|ranges| {
            let next = ranges.partition_point(|&(_, end)| end < pos);
            ranges.get(next).is_some_and(|&(start, _)| start <= pos + 1)
        })
    }

    fn value_at(&self, reg: Reg, pos: u32) -> Option<Type> {
        let ranges = self.assigned.get(reg.index() as usize)?;
        let next = ranges.partition_point(|&(_, end, _)| end < pos);
        ranges
            .get(next)
            .and_then(|&(start, _, ty)| (start <= pos + 1).then_some(ty))
    }
}

/// One instruction's temporary locations and preservation transfers. Never
/// borrow an operand location, even if its occurrence is processed later.
struct Temporaries<'a> {
    target: &'a dyn TargetRegalloc,
    live: &'a RegisterLiveness,
    inst: InstId,
    pos: u32,
    blocked: Vec<Reg>,
    restore: bool,
    frame: &'a mut StackBatch,
    slots: &'a mut HashMap<(Reg, Type), StackSlot>,
    saved: Vec<(Reg, StackSlot, Type)>,
}

impl Temporaries<'_> {
    fn take(&mut self, class: RegClass, accepts: impl Fn(Reg) -> bool) -> Result<Reg> {
        let candidates = || {
            self.target
                .spill_scratch(class)
                .iter()
                .chain(self.target.desc().allocatable_regs_in_class(class))
                .copied()
        };
        let available = |reg: Reg| {
            accepts(reg) && !self.blocked.contains(&reg) && !self.live.fixed_at(reg, self.pos)
        };
        let free =
            candidates().find(|&reg| available(reg) && self.live.value_at(reg, self.pos).is_none());
        let reg = free
            .or_else(|| {
                self.restore
                    .then(|| candidates().find(|&reg| available(reg)))
                    .flatten()
            })
            .ok_or_else(|| {
                Error::codegen(format!(
                    "no compatible {class:?} temporary at {:?}",
                    self.inst
                ))
            })?;
        if let Some(ty) = self.live.value_at(reg, self.pos) {
            // Preserve the resident value's width, not the new temporary's width.
            let slot = if let Some(&slot) = self.slots.get(&(reg, ty)) {
                slot
            } else {
                let layout = self
                    .target
                    .desc()
                    .data_layout
                    .layout_of(ty)
                    .ok_or_else(|| Error::codegen("unknown borrowed register storage layout"))?;
                let size = layout.alloc_size().ok_or_else(|| {
                    Error::codegen("borrowed register requires fixed storage size")
                })?;
                let slot =
                    self.frame
                        .alloc_object(veloc_lir::StackObject::Local, size, layout.align);
                self.slots.insert((reg, ty), slot);
                slot
            };
            self.saved.push((reg, slot, ty));
        }
        self.blocked.push(reg);
        Ok(reg)
    }
}

#[derive(Clone)]
struct Interval {
    reg: Reg,
    start: u32,
    end: u32,
    class: RegClass,
    allowed: Option<Vec<Reg>>,
}

pub struct RegisterAllocator<'a> {
    pub(super) target: &'a dyn TargetRegalloc,
    allocation: SecondaryMap<VReg, Option<PReg>>,
    spilled: SecondaryMap<VReg, Option<StackSlot>>,
}

impl<'a> RegisterAllocator<'a> {
    pub fn new(target: &'a dyn TargetRegalloc) -> Self {
        Self {
            target,
            allocation: SecondaryMap::new(),
            spilled: SecondaryMap::new(),
        }
    }

    pub(super) fn assigned(&self, reg: Reg) -> Option<PReg> {
        self.allocation.get(reg.as_vreg()?).copied().flatten()
    }

    fn assign(&mut self, reg: Reg, location: Reg) {
        self.allocation[reg.as_vreg().expect("allocation key must be virtual")] = Some(
            location
                .as_preg()
                .expect("allocation location must be physical"),
        );
    }

    pub(super) fn spill_slot(&self, reg: Reg) -> Option<StackSlot> {
        self.spilled.get(reg.as_vreg()?).copied().flatten()
    }

    pub fn allocate(
        mut self,
        mut source: MachineFunction,
        analyses: &mut FunctionAnalysisCtx,
    ) -> Result<Allocation> {
        let f = &source;
        let mut frame = f.stack_frame.batch();
        let live = analyses.liveness(f, self.target);
        let mut ranges = SecondaryMap::<VReg, Option<(u32, u32)>>::with_capacity(f.vregs().len());
        let mut fixed = Vec::<Vec<(u32, u32)>>::new();
        let mut constraints =
            SecondaryMap::<VReg, Option<Vec<Reg>>>::with_capacity(f.vregs().len());
        let mut local_fixed = Vec::<Option<(u32, u32)>>::new();
        let mut pos = 0u32;
        for block in f.blocks() {
            let start = pos * 2;
            local_fixed.fill(None);
            for &param in f.block_params(block).unwrap() {
                extend_range(&mut ranges, &mut local_fixed, param, start);
            }
            for id in f.block_insts(block) {
                let inst = &f.inst(id);
                if let veloc_lir::MachineOpcode::Target(op) = inst.opcode() {
                    for constraint in self.target.instruction_metadata(op).register_constraints {
                        let operands = if constraint.result {
                            inst.results()
                        } else {
                            inst.inputs()
                        };
                        let reg = *operands.get(constraint.operand).ok_or_else(|| {
                            Error::codegen("missing constrained register operand")
                        })?;
                        if reg.is_vreg() {
                            let allowed = constraints[reg.as_vreg().unwrap()]
                                .get_or_insert_with(|| constraint.registers.to_vec());
                            allowed.retain(|reg| constraint.registers.contains(reg));
                            if allowed.is_empty() {
                                return Err(Error::codegen(format!(
                                    "{reg:?} needs a register-class transfer between uses"
                                )));
                            }
                        } else if !constraint.registers.contains(&reg) {
                            return Err(Error::codegen(
                                "physical operand violates its register constraint",
                            ));
                        }
                    }
                }
                // All current machine schemas read inputs before writing defs.
                // Separate positions let a dying input share an output register.
                for reg in inst.uses() {
                    extend_range(&mut ranges, &mut local_fixed, reg, pos * 2);
                }
                for reg in inst.defs() {
                    extend_range(&mut ranges, &mut local_fixed, reg, pos * 2 + 1);
                }
                // Clobbers reserve a write point, not the entire block span
                // between separate calls. Uses occur one position earlier.
                for reg in inst.clobbers() {
                    let index = reg.index() as usize;
                    fixed.resize_with(fixed.len().max(index + 1), Vec::new);
                    fixed[index].push((pos * 2 + 1, pos * 2 + 1));
                }
                pos += 1;
            }
            let end = pos * 2;
            for reg in live.live_in(block).into_iter().flat_map(|set| set.iter()) {
                extend_range(&mut ranges, &mut local_fixed, reg, start);
            }
            for reg in live.live_out(block).into_iter().flat_map(|set| set.iter()) {
                extend_range(&mut ranges, &mut local_fixed, reg, end);
            }
            for (index, range) in local_fixed.iter().copied().enumerate() {
                if let Some(range) = range {
                    if fixed.len() <= index {
                        fixed.resize_with(index + 1, Vec::new);
                    }
                    fixed[index].push(range);
                }
            }
        }
        for ranges in &mut fixed {
            ranges.sort_unstable_by_key(|range| range.0);
            // Keep ends monotone for the interference binary search below.
            let mut count = 0;
            for i in 0..ranges.len() {
                let range = ranges[i];
                if count > 0 && range.0 <= ranges[count - 1].1 {
                    ranges[count - 1].1 = ranges[count - 1].1.max(range.1);
                } else {
                    ranges[count] = range;
                    count += 1;
                }
            }
            ranges.truncate(count);
        }
        let mut intervals: Vec<_> = ranges
            .iter()
            .filter_map(|(vreg, range)| range.map(|range| (vreg, range)))
            .map(|(vreg, (start, end))| {
                let reg = Reg::new_vreg(vreg.as_u32());
                let data = f.vreg_data(reg);
                Interval {
                    reg,
                    start,
                    end,
                    class: self.target.desc().reg_class_for_vreg(&data.ty, data.bank),
                    allowed: constraints[vreg].take(),
                }
            })
            .collect();
        intervals.sort_by_key(|i| (i.start, i.reg));
        let mut active = Vec::<Option<Interval>>::new();
        for interval in intervals {
            for old in &mut active {
                if old.as_ref().is_some_and(|old| old.end < interval.start) {
                    *old = None;
                }
            }
            let available = |&reg: &Reg| {
                interval
                    .allowed
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(&reg))
                    && !self.target.spill_scratch(interval.class).contains(&reg)
                    && !fixed.get(reg.index() as usize).is_some_and(|ranges| {
                        let next = ranges.partition_point(|&(_, end)| end < interval.start);
                        ranges
                            .get(next)
                            .is_some_and(|&(start, _)| start <= interval.end)
                    })
            };
            let candidates = self.target.desc().allocatable_regs_in_class(interval.class);
            let free = candidates
                .iter()
                .filter(|r| available(r))
                .find(|r| active.get(r.index() as usize).is_none_or(Option::is_none))
                .copied();
            let chosen = free.or_else(|| {
                // Evict the furthest-ending range only when the current one ends sooner.
                candidates
                    .iter()
                    .filter(|r| available(r))
                    .filter_map(|&r| {
                        active
                            .get(r.index() as usize)
                            .and_then(Option::as_ref)
                            .filter(|old| old.end > interval.end)
                            .map(|old| (r, old.end))
                    })
                    .max_by_key(|&(reg, end)| (end, reg))
                    .map(|(reg, _)| reg)
            });
            if let Some(reg) = chosen {
                let index = reg.index() as usize;
                if active.len() <= index {
                    active.resize_with(index + 1, || None);
                }
                if let Some(old) = active[index].take() {
                    self.allocation[old.reg.as_vreg().unwrap()] = None;
                    self.spill(old.reg, f, &mut frame)?;
                }
                self.assign(interval.reg, reg);
                active[index] = Some(interval);
            } else {
                self.spill(interval.reg, f, &mut frame)?;
            }
        }
        let mut physical = RegisterLiveness {
            fixed,
            assigned: Vec::new(),
        };
        for (vreg, range) in ranges.iter() {
            if let (Some(preg), Some((start, end))) = (self.allocation[vreg], *range) {
                let reg: Reg = preg.into();
                let index = reg.index() as usize;
                physical
                    .assigned
                    .resize_with(physical.assigned.len().max(index + 1), Vec::new);
                physical.assigned[index].push((
                    start,
                    end,
                    source.vreg_data(Reg::new_vreg(vreg.as_u32())).ty,
                ));
            }
        }
        for ranges in &mut physical.assigned {
            ranges.sort_unstable_by_key(|&(start, _, _)| start);
        }
        let instructions = self.plan(&source, &mut frame, &physical)?;
        let edges = self.plan_edges(&mut source, &mut frame)?;
        Ok(Allocation {
            source,
            instructions,
            edges,
            frame,
        })
    }

    fn spill(&mut self, reg: Reg, f: &MachineFunction, frame: &mut StackBatch) -> Result<()> {
        let ty = f.vreg_data(reg).ty;
        let layout = &self.target.desc().data_layout;
        self.target
            .desc()
            .registers
            .special_regs
            .frame_pointer
            .ok_or_else(|| Error::codegen("spilling requires a frame pointer"))?;
        let layout = layout
            .layout_of(ty)
            .ok_or_else(|| Error::codegen(format!("unknown storage layout: {ty:?}")))?;
        let size = layout.alloc_size().ok_or_else(|| {
            Error::codegen(format!("stack allocation requires fixed size: {ty:?}"))
        })?;
        let align = layout.align;
        let slot = frame.alloc_object(veloc_lir::StackObject::Local, size, align);
        self.spilled[reg.as_vreg().expect("spill key must be virtual")] = Some(slot);
        Ok(())
    }

    fn plan(
        &self,
        f: &MachineFunction,
        frame: &mut StackBatch,
        live: &RegisterLiveness,
    ) -> Result<SecondaryMap<InstId, InstAllocation>> {
        let mut instructions = SecondaryMap::new();
        let mut slots = HashMap::new();
        let mut pos = 0;
        let mut block = f.blocks().next();
        while let Some(current_block) = block {
            let next_block = f.layout().next_block(current_block);
            let mut cursor = f.layout().first_inst(current_block);
            while let Some(id) = cursor {
                let next_id = f.layout().next_inst(id);
                {
                    let inst = &f.inst(id);
                    let ties = match inst.opcode() {
                        veloc_lir::MachineOpcode::Target(op) => {
                            self.target.instruction_metadata(op).tied_operands
                        }
                        _ => &[],
                    };
                    let register_constraints = match inst.opcode() {
                        veloc_lir::MachineOpcode::Target(op) => {
                            self.target.instruction_metadata(op).register_constraints
                        }
                        _ => &[],
                    };
                    let accepts = |result: bool, operand: usize, reg: Reg| {
                        register_constraints
                            .iter()
                            .filter(|constraint| {
                                constraint.result == result && constraint.operand == operand
                            })
                            .all(|constraint| constraint.registers.contains(&reg))
                    };
                    let accepts_value = |value: Reg, location: Reg| {
                        inst.inputs()
                            .iter()
                            .enumerate()
                            .all(|(i, &reg)| reg != value || accepts(false, i, location))
                            && inst
                                .results()
                                .iter()
                                .enumerate()
                                .all(|(i, &reg)| reg != value || accepts(true, i, location))
                    };
                    if ties.len() > 1 {
                        return Err(Error::codegen(
                            "multiple output reuse constraints require parallel allocation edits",
                        ));
                    }
                    let mut plan = InstAllocation::default();
                    let blocked = inst
                        .uses()
                        .chain(inst.defs())
                        .chain(inst.clobbers())
                        .filter_map(|reg| reg.as_preg().or_else(|| self.assigned(reg)))
                        .map(Reg::from)
                        .collect();
                    let mut temps = Temporaries {
                        target: self.target,
                        live,
                        inst: id,
                        pos,
                        blocked,
                        // Restores after branches/returns would not execute on all paths.
                        restore: matches!(
                            self.target.control_flow(inst),
                            veloc_lir::ControlFlow::Next | veloc_lir::ControlFlow::Call
                        ),
                        frame,
                        slots: &mut slots,
                        saved: Vec::new(),
                    };
                    let mut bindings = HashMap::new();
                    let mut loads = Vec::new();
                    let mut stores = Vec::new();
                    let result_count = inst.results().len();
                    let fields = inst
                        .results()
                        .iter()
                        .copied()
                        .map(|r| (r, true))
                        .chain(inst.inputs().iter().copied().map(|r| (r, false)));
                    for (index, (reg, write)) in fields.enumerate() {
                        let read = !write;
                        if reg.is_preg() {
                            if write {
                                plan.results.push(reg.as_preg().unwrap());
                            } else {
                                plan.locations.push(reg.as_preg().unwrap());
                            }
                            continue;
                        }
                        let preg = if let Some(preg) = self.assigned(reg) {
                            preg.into()
                        } else {
                            let slot = self
                                .spill_slot(reg)
                                .ok_or_else(|| Error::codegen("unallocated virtual register"))?;
                            let data = f.vreg_data(reg);
                            let ty = data.ty;
                            let class = self.target.desc().reg_class_for_vreg(&ty, data.bank);
                            let tied_spill = ties
                                .iter()
                                .find(|tie| tie.use_operand + result_count == index)
                                .map(|tie| inst.results()[tie.result])
                                .filter(|dst| self.spill_slot(*dst).is_some())
                                .and_then(|dst| bindings.get(&dst).copied());
                            let preg =
                                if let Some(preg) = bindings.get(&reg).copied().or(tied_spill) {
                                    bindings.insert(reg, preg);
                                    preg
                                } else {
                                    // Inputs are read before results are written. An
                                    // untied spilled result may reuse one reload's
                                    // location, but two distinct inputs may not.
                                    let reuse = read
                                        .then(|| {
                                            inst.results()
                                                .iter()
                                                .enumerate()
                                                .filter(|(result, _)| {
                                                    !ties.iter().any(|tie| tie.result == *result)
                                                })
                                                .filter_map(|(_, dst)| bindings.get(dst).copied())
                                                .find(|r| {
                                                    accepts_value(reg, *r)
                                                        && self
                                                            .target
                                                            .desc()
                                                            .registers
                                                            .reg_class(class)
                                                            .is_some_and(|info| {
                                                                info.members.contains(r)
                                                            })
                                                        && !loads
                                                            .iter()
                                                            .any(|&(_, loaded, _)| loaded == *r)
                                                })
                                        })
                                        .flatten();
                                    let preg = if let Some(reg) = reuse {
                                        reg
                                    } else {
                                        temps.take(class, |location| {
                                            accepts_value(reg, location)
                                                && ties.iter().all(|tie| {
                                                    !write
                                                        || tie.result != index
                                                        || accepts_value(
                                                            inst.inputs()[tie.use_operand],
                                                            location,
                                                        )
                                                })
                                        })?
                                    };
                                    bindings.insert(reg, preg);
                                    preg
                                };
                            if read && !loads.iter().any(|&(s, _, _)| s == slot) {
                                loads.push((slot, preg, ty));
                            }
                            if write && !stores.iter().any(|&(s, _, _)| s == slot) {
                                stores.push((slot, preg, ty));
                            }
                            preg
                        };
                        let preg = preg
                            .as_preg()
                            .expect("allocator must assign physical registers");
                        if write {
                            plan.results.push(preg);
                        } else {
                            plan.locations.push(preg);
                        }
                    }
                    // Resolve tied locations without changing virtual identities.
                    // A dying unrelated input may share the output's assigned register;
                    // use a scratch in that case, so the pre-copy cannot destroy it.
                    let mut copies_before = Vec::new();
                    let mut copies_after = Vec::new();
                    for tie in ties {
                        let dst = inst
                            .results()
                            .get(tie.result)
                            .copied()
                            .ok_or_else(|| Error::codegen("missing tied definition"))?;
                        let input_index = tie.use_operand;
                        let input = inst.inputs()[input_index];
                        let output_location = plan.results[tie.result];
                        let input_location = plan.locations[input_index];
                        if output_location == input_location {
                            continue;
                        }
                        let ty = if dst.is_vreg() {
                            f.vreg_data(dst).ty
                        } else if input.is_vreg() {
                            f.vreg_data(input).ty
                        } else {
                            return Err(Error::codegen(
                                "physical tied operands must already agree",
                            ));
                        };
                        let conflicts =
                            plan.locations.iter().enumerate().any(|(index, &location)| {
                                index != input_index && location == output_location
                            });
                        let work = if conflicts {
                            let data = f.vreg_data(if dst.is_vreg() { dst } else { input });
                            let class = self.target.desc().reg_class_for_vreg(&ty, data.bank);
                            temps
                                .take(class, |reg| {
                                    accepts(true, tie.result, reg)
                                        && accepts(false, input_index, reg)
                                })?
                                .as_preg()
                                .expect("physical temporary")
                        } else {
                            output_location
                        };
                        copies_before.push((work.into(), input_location.into(), ty));
                        if work != output_location {
                            copies_after.push((output_location.into(), work.into(), ty));
                        }
                        plan.results[tie.result] = work;
                        plan.locations[input_index] = work;
                    }
                    // Preservation encloses all reloads, copies and result stores.
                    for &(reg, slot, ty) in &temps.saved {
                        plan.before.push(Transfer::Spill {
                            kind: SpillKind::Store,
                            reg,
                            slot,
                            ty,
                        });
                    }
                    // Reloads precede input copies; output copies precede spill stores.
                    for (dst, src, ty) in copies_after {
                        plan.after.push(Transfer::Copy { dst, src, ty });
                    }
                    for (load, accesses) in [(true, loads), (false, stores)] {
                        for (slot, reg, ty) in accesses {
                            let inst = Transfer::Spill {
                                kind: if load {
                                    SpillKind::Load
                                } else {
                                    SpillKind::Store
                                },
                                reg,
                                slot,
                                ty,
                            };
                            if load {
                                plan.before.push(inst);
                            } else {
                                plan.after.push(inst);
                            }
                        }
                    }
                    for (dst, src, ty) in copies_before {
                        plan.before.push(Transfer::Copy { dst, src, ty });
                    }
                    for (reg, slot, ty) in temps.saved {
                        plan.after.push(Transfer::Spill {
                            kind: SpillKind::Load,
                            reg,
                            slot,
                            ty,
                        });
                    }
                    instructions[id] = plan;
                }
                pos += 2;
                cursor = next_id;
            }
            block = next_block;
        }
        Ok(instructions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::TargetInfo;
    use crate::target::x86_64::{
        X86_64TargetMachine,
        inst::{REG_RAX, REG_RCX, REG_RDX, TargetInst},
    };
    use veloc_lir::Type;

    #[test]
    fn three_spilled_inputs_use_free_or_preserved_registers() {
        let target = X86_64TargetMachine::new(crate::TargetConfig::default()).unwrap();
        for occupied in [false, true] {
            let mut f = MachineFunction::new("indexed_store".into());
            let src = f.editor().alloc_vreg(Type::I64);
            let base = f.editor().alloc_vreg(Type::PTR);
            let index = f.editor().alloc_vreg(Type::PTR);
            let mut ids = Vec::new();
            for _ in 0..2 {
                let block = f.entry_block();
                ids.push(TargetInst::X86Store64Index.write(
                    f.editor().at_end(block).writer(),
                    &[],
                    &[src, base, index],
                    [veloc_lir::FieldValue::Imm(16)],
                ));
            }
            let mut allocator = RegisterAllocator::new(&target);
            let mut frame = f.stack_frame.batch();
            for value in [src, base, index] {
                allocator.spill(value, &f, &mut frame).unwrap();
            }
            let mut live = RegisterLiveness::default();
            if occupied {
                for &reg in target.desc().allocatable_regs_in_class(RegClass::GPR) {
                    if target.spill_scratch(RegClass::GPR).contains(&reg) {
                        continue;
                    }
                    let index = reg.index() as usize;
                    live.assigned
                        .resize_with(live.assigned.len().max(index + 1), Vec::new);
                    live.assigned[index].push((0, 100, Type::I64));
                }
            }
            let plans = allocator.plan(&f, &mut frame, &live).unwrap();
            // Repeated borrowing reuses the preservation slot.
            assert_eq!(frame.slots().len(), 3 + usize::from(occupied));
            for id in ids {
                let plan = &plans[id];
                let mut locations = plan.locations.to_vec();
                locations.sort_unstable();
                locations.dedup();
                assert_eq!(locations.len(), 3);
                assert_eq!(plan.before.len(), 3 + usize::from(occupied));
                assert_eq!(plan.after.len(), usize::from(occupied));
                if occupied {
                    let Transfer::Spill {
                        kind: SpillKind::Store,
                        reg,
                        slot,
                        ty,
                    } = plan.before[0]
                    else {
                        panic!("save before reloads")
                    };
                    let Transfer::Spill {
                        kind: SpillKind::Load,
                        reg: restored,
                        slot: saved,
                        ty: saved_ty,
                    } = plan.after[0]
                    else {
                        panic!("restore after instruction")
                    };
                    assert_eq!((reg, slot, ty), (restored, saved, saved_ty));
                    assert_eq!(ty, Type::I64);
                }
            }
        }
    }

    #[test]
    fn tied_allocation_preserves_inputs_with_collisions_and_spills() {
        let target = X86_64TargetMachine::new(crate::TargetConfig::default()).unwrap();
        for mode in 0..3 {
            let mut f = MachineFunction::new("reuse".into());
            let lhs = f.editor().alloc_vreg(Type::I64);
            let rhs = f.editor().alloc_vreg(Type::I64);
            let dst = f.editor().alloc_vreg(Type::I64);
            let id = TargetInst::X86Sub64.write(
                f.editor().at_end(veloc_lir::BlockId::from_u32(0)).writer(),
                &[dst],
                &[rhs, lhs],
                [],
            );

            let mut allocator = RegisterAllocator::new(&target);
            let mut frame = f.stack_frame.batch();
            if mode == 2 {
                for reg in [lhs, rhs, dst] {
                    allocator.spill(reg, &f, &mut frame).unwrap();
                }
            } else {
                allocator.assign(lhs, REG_RAX);
                allocator.assign(rhs, REG_RCX);
                allocator.assign(dst, if mode == 0 { REG_RDX } else { REG_RCX });
            }
            let instructions = allocator
                .plan(&f, &mut frame, &RegisterLiveness::default())
                .unwrap();
            let plan = &instructions[id];
            assert_eq!(plan.results[0], plan.locations[1]);
            assert_ne!(plan.results[0], plan.locations[0]);
            assert!(!plan.before.is_empty());
            if mode == 1 {
                assert_eq!(plan.after.len(), 1);
            }
            if mode == 2 {
                assert_eq!((plan.before.len(), plan.after.len()), (2, 1));
            }
            assert_eq!(f.inst(id).defs().collect::<Vec<_>>(), [dst]);
            assert_eq!(f.inst(id).uses().collect::<Vec<_>>(), [rhs, lhs]);
            f.check_refs().unwrap();
            let physical = Allocation {
                source: f,
                instructions,
                edges: Vec::new(),
                frame,
            }
            .materialize(&target)
            .unwrap();
            assert_eq!(
                Some(physical.inst(id).results()[0]),
                physical.inst(id).inputs().get(1).copied()
            );
            physical.check_refs().unwrap();
        }
    }
}

fn extend_range(
    virtual_: &mut SecondaryMap<VReg, Option<(u32, u32)>>,
    physical: &mut Vec<Option<(u32, u32)>>,
    reg: Reg,
    pos: u32,
) {
    let range = if let Some(reg) = reg.as_vreg() {
        &mut virtual_[reg]
    } else {
        let index = reg.index() as usize;
        if physical.len() <= index {
            physical.resize(index + 1, None);
        }
        &mut physical[index]
    };
    let range = range.get_or_insert((pos, pos));
    range.0 = range.0.min(pos);
    range.1 = range.1.max(pos);
}
