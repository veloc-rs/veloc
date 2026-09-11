//! Global linear scan with CFG liveness, fixed registers and whole-range spills.
use crate::pipeline::FunctionAnalysisCtx;
use crate::target::arch::{CallConv, RegClass, TargetMachine};
use crate::{Error, Result};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use veloc_lir::{InstExtra, MachineFunction, MachineOperand, Reg, StackSlot, Writable};

#[derive(Clone)]
struct Interval {
    reg: Reg,
    start: u32,
    end: u32,
    class: RegClass,
}

pub struct RegisterAllocator<'a> {
    target: &'a dyn TargetMachine,
    allocation: BTreeMap<Reg, Reg>,
    spilled: BTreeMap<Reg, StackSlot>,
}

impl<'a> RegisterAllocator<'a> {
    pub fn new(target: &'a dyn TargetMachine) -> Self {
        Self {
            target,
            allocation: BTreeMap::new(),
            spilled: BTreeMap::new(),
        }
    }

    pub fn allocate<S>(
        &mut self,
        f: &mut MachineFunction<S>,
        cc: veloc_mir::CallConv,
        analyses: &mut FunctionAnalysisCtx,
    ) -> Result<()> {
        self.allocation.clear();
        self.spilled.clear();
        let live = analyses.liveness(f, self.target);
        let mut ranges: BTreeMap<Reg, (u32, u32)> = BTreeMap::new();
        let mut fixed: BTreeMap<Reg, Vec<(u32, u32)>> = BTreeMap::new();
        let mut calls = Vec::new();
        let mut pos = 0u32;
        for block in &f.blocks {
            let start = pos * 2;
            let end = (pos + block.insts.len() as u32) * 2;
            let mut local = BTreeMap::new();
            for &id in &block.insts {
                let inst = &f.dfg[id];
                // All current machine schemas read inputs before writing defs.
                // Separate positions let a dying input share an output register.
                for reg in inst.uses() {
                    extend(&mut local, reg, pos * 2);
                }
                for reg in inst.defs() {
                    extend(&mut local, reg, pos * 2 + 1);
                }
                if self.target.is_call(inst) || matches!(f.inst_extra(id), Some(InstExtra::Call(_)))
                {
                    calls.push(pos * 2 + 1);
                }
                pos += 1;
            }
            for &reg in live.live_in(block.id).into_iter().flatten() {
                extend(&mut local, reg, start);
            }
            for &reg in live.live_out(block.id).into_iter().flatten() {
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
                (!crosses_call || preserved.contains(&reg))
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
                    self.spill(old.reg, f)?;
                }
                self.allocation.insert(interval.reg, reg);
                active.insert(reg, interval);
            } else {
                self.spill(interval.reg, f)?;
            }
        }
        self.rewrite(f)?;
        f.is_regallocated = true;
        Ok(())
    }

    fn spill<S>(&mut self, reg: Reg, f: &mut MachineFunction<S>) -> Result<()> {
        let ty = f.vreg_data(reg).ty;
        let layout = &self.target.desc().data_layout;
        self.target
            .desc()
            .registers
            .special_regs
            .frame_pointer
            .ok_or_else(|| Error::codegen("spilling requires a frame pointer"))?;
        let slot = f.alloc_stack_slot(layout.type_size(&ty), layout.type_align(&ty));
        self.spilled.insert(reg, slot);
        Ok(())
    }

    fn rewrite<S>(&self, f: &mut MachineFunction<S>) -> Result<()> {
        for block in 0..f.num_blocks() {
            f.rewrite_block(block, |cursor| {
                let mut inst = cursor.current_inst_clone();
                let extra = cursor.current_extra_cloned();
                let mut occupied: Vec<_> = inst.uses().filter(|r| !r.is_vreg()).collect();
                if !self.target.is_call(&inst) {
                    occupied.extend(inst.defs().filter(|r| !r.is_vreg()));
                }
                let mut bindings = BTreeMap::new();
                let mut loads = Vec::new();
                let mut stores = Vec::new();
                for operand in &mut inst.operands {
                    let (reg, read, write) = match *operand {
                        MachineOperand::Use(r) => (r, true, false),
                        MachineOperand::Def(w) => (w.to_reg(), false, true),
                        MachineOperand::TiedDefUse(w) => (w.to_reg(), true, true),
                        _ => continue,
                    };
                    if !reg.is_vreg() {
                        continue;
                    }
                    let preg = if let Some(&preg) = self.allocation.get(&reg) {
                        preg
                    } else {
                        let slot = *self
                            .spilled
                            .get(&reg)
                            .ok_or_else(|| Error::codegen("unallocated virtual register"))?;
                        let data = cursor.mfunc().vreg_data(reg);
                        let ty = data.ty;
                        let class = self.target.desc().reg_class_for_vreg(&ty, data.bank);
                        let preg = if let Some(&preg) = bindings.get(&reg) {
                            preg
                        } else {
                            let preg = self
                                .target
                                .spill_scratch(class)
                                .iter()
                                .copied()
                                .find(|r| {
                                    !occupied.contains(r) && !bindings.values().any(|s| s == r)
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
                    *operand = match (read, write) {
                        (true, true) => MachineOperand::TiedDefUse(Writable(preg)),
                        (true, false) => MachineOperand::Use(preg),
                        _ => MachineOperand::Def(Writable(preg)),
                    };
                }
                for (slot, reg, ty) in loads {
                    let slot = &cursor.mfunc().stack_frame.slots[slot];
                    cursor.emit_before(
                        self.target.spill_instruction(
                            true,
                            reg,
                            slot.base.resolve(
                                self.target
                                    .desc()
                                    .registers
                                    .special_regs
                                    .frame_pointer
                                    .unwrap(),
                            ),
                            slot.offset as i64,
                            ty,
                        )?,
                    );
                }
                cursor.replace_current(inst);
                if let Some(extra) = extra {
                    cursor.set_current_extra(extra);
                }
                for (slot, reg, ty) in stores {
                    let slot = &cursor.mfunc().stack_frame.slots[slot];
                    cursor.emit_before(
                        self.target.spill_instruction(
                            false,
                            reg,
                            slot.base.resolve(
                                self.target
                                    .desc()
                                    .registers
                                    .special_regs
                                    .frame_pointer
                                    .unwrap(),
                            ),
                            slot.offset as i64,
                            ty,
                        )?,
                    );
                }
                Ok::<(), Error>(())
            })?;
        }
        Ok(())
    }

    pub fn get_allocation(&self, reg: Reg) -> Option<Reg> {
        self.allocation.get(&reg).copied()
    }
    pub fn get_stack_slot(&self, reg: Reg) -> Option<StackSlot> {
        self.spilled.get(&reg).copied()
    }
}

fn extend(ranges: &mut BTreeMap<Reg, (u32, u32)>, reg: Reg, pos: u32) {
    let range = ranges.entry(reg).or_insert((pos, pos));
    range.0 = range.0.min(pos);
    range.1 = range.1.max(pos);
}
