//! Global linear scan with CFG liveness, fixed registers and whole-range spills.
use super::allocation::{Allocation, InstAllocation};
use crate::pipeline::FunctionAnalysisCtx;
use crate::target::arch::{CallConv, RegClass, SpillKind, TargetRegalloc};
use crate::{Error, Result};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;
use cranelift_entity::SecondaryMap;
use veloc_lir::{InstId, MachineFunction, Reg, StackFrame, StackSlot};

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
    pub(super) allocation: BTreeMap<Reg, Reg>,
    pub(super) spilled: BTreeMap<Reg, StackSlot>,
}

impl<'a> RegisterAllocator<'a> {
    pub fn new(target: &'a dyn TargetRegalloc) -> Self {
        Self {
            target,
            allocation: BTreeMap::new(),
            spilled: BTreeMap::new(),
        }
    }

    pub fn allocate(
        mut self,
        mut source: MachineFunction,
        cc: veloc_mir::CallConv,
        analyses: &mut FunctionAnalysisCtx,
    ) -> Result<Allocation> {
        let f = &source;
        let mut frame = f.stack_frame.clone();
        let live = analyses.liveness(f, self.target);
        let mut ranges: BTreeMap<Reg, (u32, u32)> = BTreeMap::new();
        let mut fixed: BTreeMap<Reg, Vec<(u32, u32)>> = BTreeMap::new();
        let mut calls = Vec::new();
        let mut constraints: BTreeMap<Reg, Vec<Reg>> = BTreeMap::new();
        let mut pos = 0u32;
        for block in f.blocks() {
            let start = pos * 2;
            let end = (pos + f.block_insts(block).count() as u32) * 2;
            let mut local = BTreeMap::new();
            for &param in f.block_params(block).unwrap() {
                extend(&mut local, param, start);
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
                            let allowed = constraints
                                .entry(reg)
                                .or_insert_with(|| constraint.registers.to_vec());
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
                    extend(&mut local, reg, pos * 2);
                }
                for reg in inst.defs() {
                    extend(&mut local, reg, pos * 2 + 1);
                }
                if self.target.is_call(inst)
                    || matches!(f.inst_extra(id), Some(veloc_lir::InstExtraRef::Call(_)))
                {
                    calls.push(pos * 2 + 1);
                }
                pos += 1;
            }
            for &reg in live.live_in(block).into_iter().flatten() {
                extend(&mut local, reg, start);
            }
            for &reg in live.live_out(block).into_iter().flatten() {
                extend(&mut local, reg, end);
            }
            for (reg, (start, end)) in local {
                if reg.is_vreg() {
                    extend(&mut ranges, reg, start);
                    extend(&mut ranges, reg, end);
                } else {
                    fixed.entry(reg).or_default().push((start, end));
                }
            }
        }
        let mut intervals: Vec<_> = ranges
            .into_iter()
            .map(|(reg, (start, end))| {
                let data = f.vreg_data(reg);
                Interval {
                    reg,
                    start,
                    end,
                    class: self.target.desc().reg_class_for_vreg(&data.ty, data.bank),
                    allowed: constraints.remove(&reg),
                }
            })
            .collect();
        intervals.sort_by_key(|i| (i.start, i.reg));
        let preserved = CallConv::from(cc).preserved_regs(self.target.desc().arch);
        let mut active: BTreeMap<Reg, Interval> = BTreeMap::new();
        for interval in intervals {
            active.retain(|_, old| old.end >= interval.start);
            let crosses_call = calls
                .iter()
                .any(|&p| interval.start < p && p < interval.end);
            let available = |&reg: &Reg| {
                interval
                    .allowed
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(&reg))
                    && (!crosses_call || preserved.contains(&reg))
                    && !self.target.spill_scratch(interval.class).contains(&reg)
                    && !fixed.get(&reg).is_some_and(|rs| {
                        rs.iter()
                            .any(|&(s, e)| s <= interval.end && interval.start <= e)
                    })
            };
            let candidates = self.target.desc().allocatable_regs_in_class(interval.class);
            let free = candidates
                .iter()
                .filter(|r| available(r))
                .find(|r| !active.contains_key(r))
                .copied();
            let chosen = free.or_else(|| {
                // Evict the furthest-ending range only when the current one ends sooner.
                candidates
                    .iter()
                    .filter(|r| available(r))
                    .filter_map(|&r| {
                        active
                            .get(&r)
                            .filter(|old| old.end > interval.end)
                            .map(|old| (r, old.end))
                    })
                    .max_by_key(|&(reg, end)| (end, reg))
                    .map(|(reg, _)| reg)
            });
            if let Some(reg) = chosen {
                if let Some(old) = active.remove(&reg) {
                    self.allocation.remove(&old.reg);
                    self.spill(old.reg, f, &mut frame)?;
                }
                self.allocation.insert(interval.reg, reg);
                active.insert(reg, interval);
            } else {
                self.spill(interval.reg, f, &mut frame)?;
            }
        }
        let instructions = self.plan(&mut source, &frame)?;
        let edges = self.plan_edges(&mut source, &mut frame)?;
        Ok(Allocation {
            source,
            instructions,
            edges,
            frame,
        })
    }

    fn spill(&mut self, reg: Reg, f: &MachineFunction, frame: &mut StackFrame) -> Result<()> {
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
        let slot = frame.alloc_slot(size, align);
        self.spilled.insert(reg, slot);
        Ok(())
    }

    fn plan(
        &self,
        f: &mut MachineFunction,
        frame: &StackFrame,
    ) -> Result<SecondaryMap<InstId, InstAllocation>> {
        let mut instructions = SecondaryMap::new();
        let layout: Vec<_> = f.blocks().flat_map(|b| f.block_insts(b)).collect();
        for id in layout {
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
                if ties.len() > 1 {
                    return Err(Error::codegen(
                        "multiple output reuse constraints require parallel allocation edits",
                    ));
                }
                let mut plan = InstAllocation::default();
                let mut occupied: Vec<_> = inst.uses().filter(|r| r.is_preg()).collect();
                if !self.target.is_call(inst) {
                    occupied.extend(inst.defs().filter(|r| r.is_preg()));
                }
                let mut bindings = BTreeMap::new();
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
                        plan.locations.push(reg.as_preg().unwrap());
                        continue;
                    }
                    let preg = if let Some(&preg) = self.allocation.get(&reg) {
                        preg
                    } else {
                        let slot = *self
                            .spilled
                            .get(&reg)
                            .ok_or_else(|| Error::codegen("unallocated virtual register"))?;
                        let data = f.vreg_data(reg);
                        let ty = data.ty;
                        let class = self.target.desc().reg_class_for_vreg(&ty, data.bank);
                        let tied_spill = ties
                            .iter()
                            .find(|tie| tie.use_operand + result_count == index)
                            .map(|tie| inst.results()[tie.result])
                            .filter(|dst| self.spilled.contains_key(dst))
                            .and_then(|dst| bindings.get(&dst).copied());
                        let preg = if let Some(preg) = bindings.get(&reg).copied().or(tied_spill) {
                            bindings.insert(reg, preg);
                            preg
                        } else {
                            let preg = self
                                .target
                                .spill_scratch(class)
                                .iter()
                                .copied()
                                .find(|r| {
                                    accepts(
                                        write,
                                        if write { index } else { index - result_count },
                                        *r,
                                    ) && !occupied.contains(r)
                                        && !bindings.values().any(|s| s == r)
                                })
                                .ok_or_else(|| {
                                    Error::codegen(
                                        "insufficient dedicated spill temporaries for instruction",
                                    )
                                })?;
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
                    plan.locations.push(
                        preg.as_preg()
                            .expect("allocator must assign physical registers"),
                    );
                }
                plan.results = plan.locations.drain(..result_count).collect();
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
                        return Err(Error::codegen("physical tied operands must already agree"));
                    };
                    let conflicts = plan.locations.iter().enumerate().any(|(index, &location)| {
                        index != input_index && location == output_location
                    });
                    let work = if conflicts {
                        let data = f.vreg_data(if dst.is_vreg() { dst } else { input });
                        let class = self.target.desc().reg_class_for_vreg(&ty, data.bank);
                        self.target
                            .spill_scratch(class)
                            .iter()
                            .filter_map(|r| r.as_preg())
                            .find(|r| {
                                accepts(true, tie.result, (*r).into())
                                    && accepts(false, input_index, (*r).into())
                                    && !plan.locations.contains(r)
                                    && !plan.results.contains(r)
                            })
                            .ok_or_else(|| {
                                Error::codegen("insufficient temporary for tied output")
                            })?
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
                // Reloads precede input copies; output copies precede spill stores.
                for (dst, src, ty) in copies_after {
                    plan.after.push(self.target.copy_instruction(
                        f.editor().writer(),
                        dst,
                        src,
                        ty,
                    )?);
                }
                for (load, accesses) in [(true, loads), (false, stores)] {
                    for (slot, reg, ty) in accesses {
                        let slot = &frame.slots[slot];
                        let base = slot.base.resolve(
                            self.target
                                .desc()
                                .registers
                                .special_regs
                                .frame_pointer
                                .ok_or_else(|| {
                                    Error::codegen("spilling requires a frame pointer")
                                })?,
                        );
                        let inst = self.target.spill_instruction(
                            f.editor().writer(),
                            if load {
                                SpillKind::Load
                            } else {
                                SpillKind::Store
                            },
                            reg,
                            base,
                            slot.offset as i64,
                            ty,
                        )?;
                        if load {
                            plan.before.push(inst);
                        } else {
                            plan.after.push(inst);
                        }
                    }
                }
                for (dst, src, ty) in copies_before {
                    plan.before.push(self.target.copy_instruction(
                        f.editor().writer(),
                        dst,
                        src,
                        ty,
                    )?);
                }
                instructions[id] = plan;
            }
        }
        Ok(instructions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::x86_64::{
        X86_64TargetMachine,
        isle::{REG_RAX, REG_RCX, REG_RDX, TargetInst},
    };
    use veloc_lir::{MachineOpcode, Type, Writable};

    #[test]
    fn tied_allocation_preserves_inputs_with_collisions_and_spills() {
        let target = X86_64TargetMachine::new(crate::TargetConfig::default());
        for mode in 0..3 {
            let mut f = MachineFunction::new("reuse".into());
            f.editor().create_block();
            let lhs = f.editor().alloc_vreg(Type::I64);
            let rhs = f.editor().alloc_vreg(Type::I64);
            let dst = f.editor().alloc_vreg(Type::I64);
            let id = f.editor().writer().binary(
                MachineOpcode::Target(TargetInst::X86Sub64.as_u32()),
                Writable(dst),
                rhs,
                lhs,
            );
            f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
            let mut allocator = RegisterAllocator::new(&target);
            let mut frame = f.stack_frame.clone();
            if mode == 2 {
                for reg in [lhs, rhs, dst] {
                    allocator.spill(reg, &f, &mut frame).unwrap();
                }
            } else {
                allocator.allocation.extend([
                    (lhs, REG_RAX),
                    (rhs, REG_RCX),
                    (dst, if mode == 0 { REG_RDX } else { REG_RCX }),
                ]);
            }
            let instructions = allocator.plan(&mut f, &frame).unwrap();
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
            .materialize();
            assert_eq!(
                Some(physical.inst(id).results()[0]),
                physical.inst(id).inputs().get(1).copied()
            );
            physical.check_refs().unwrap();
        }
    }
}

fn extend(ranges: &mut BTreeMap<Reg, (u32, u32)>, reg: Reg, pos: u32) {
    let range = ranges.entry(reg).or_insert((pos, pos));
    range.0 = range.0.min(pos);
    range.1 = range.1.max(pos);
}
