//! Global linear scan with CFG liveness, fixed registers and whole-range spills.
#[cfg(test)]
use super::allocation::Transfer;
use super::allocation::{Allocation, InstAllocation};
use crate::analysis::FunctionAnalysisCtx;
#[cfg(test)]
use crate::target::SpillKind;
use crate::target::{RegClass, TargetRegalloc};
use crate::{Error, Result};
use cranelift_entity::SecondaryMap;
use std::format;
use std::vec::Vec;
use veloc_lir::{InstId, MachineFunction, PReg, Reg, StackBatch, StackSlot, Type, VReg};

#[derive(Clone, Copy)]
pub(super) struct Reservation {
    pub start: u32,
    pub end: u32,
    pub value: Option<Reg>,
}

#[derive(Clone, Copy)]
pub(super) struct Resident {
    pub start: u32,
    pub end: u32,
    pub value: Reg,
    pub ty: Type,
}

#[derive(Default)]
pub(super) struct RegisterLiveness {
    pub fixed: Vec<Vec<Reservation>>,
    pub assigned: Vec<Vec<Resident>>,
}

impl RegisterLiveness {
    pub fn resident(&self, reg: Reg, pos: u32) -> Option<Resident> {
        let ranges = self.assigned.get(reg.index() as usize)?;
        let next = ranges.partition_point(|r| r.end < pos);
        ranges.get(next).copied().filter(|r| r.start <= pos)
    }

    pub fn reserved(&self, reg: Reg, pos: u32, value: Reg) -> bool {
        self.fixed.get(reg.index() as usize).is_some_and(|ranges| {
            ranges[ranges.partition_point(|r| r.end < pos)..]
                .iter()
                .any(|r| r.start <= pos && r.value != Some(value))
        })
    }
}

#[derive(Clone)]
struct Interval {
    reg: Reg,
    start: u32,
    end: u32,
    class: RegClass,
    preferences: Vec<Reg>,
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
        let mut fixed = Vec::<Vec<Reservation>>::new();
        let mut preferences = SecondaryMap::<VReg, Vec<Reg>>::with_capacity(f.vregs().len());
        let mut local_fixed = Vec::<Option<(u32, u32)>>::new();
        crate::verify::verify_entry_bindings(f)?;
        for binding in f.entry_bindings() {
            reserve(&mut fixed, binding.location.into(), 0, Some(binding.value));
            preferences[binding.value.as_vreg().unwrap()].push(binding.location.into());
        }
        let mut pos = 1u32;
        for block in f.blocks() {
            let start = pos * 2;
            local_fixed.fill(None);
            for &param in f.block_params(block).unwrap() {
                let definition = if block == f.entry_block() { 0 } else { start };
                extend_range(&mut ranges, &mut local_fixed, param, definition);
            }
            for id in f.block_insts(block) {
                let inst = &f.inst(id);
                super::constraints::validate(*inst, self.target, false)?;
                let requirements: Vec<_> =
                    super::constraints::constraints(*inst, self.target).collect();
                for constraint in &requirements {
                    let value = *constraint
                        .operand
                        .get(inst.inputs(), inst.results())
                        .unwrap();
                    let at = pos * 2
                        + u32::from(matches!(
                            constraint.operand,
                            veloc_lir::OperandRef::Result(_)
                        ));
                    match constraint.placement {
                        veloc_lir::Placement::Fixed(reg) | veloc_lir::Placement::State(reg) => {
                            reserve(&mut fixed, reg, at, value.as_vreg().map(|_| value));
                            if let Some(v) = value.as_vreg() {
                                preferences[v].push(reg);
                            }
                        }
                        veloc_lir::Placement::Registers(regs) => {
                            if let Some(v) = value.as_vreg() {
                                preferences[v].extend_from_slice(regs);
                            }
                        }
                        veloc_lir::Placement::Reuse(_) => {}
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
                    // A fixed result defines the new value in this register;
                    // the ABI clobber mask describes destruction of the old one.
                    if !requirements.iter().any(|c| {
                        matches!(c.operand, veloc_lir::OperandRef::Result(_))
                            && c.placement == veloc_lir::Placement::Fixed(reg)
                    }) {
                        reserve(&mut fixed, reg, pos * 2 + 1, None);
                    }
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
                    fixed[index].push(Reservation {
                        start: range.0,
                        end: range.1,
                        value: None,
                    });
                }
            }
        }
        for reservations in &mut fixed {
            reservations.sort_unstable_by_key(|r| r.end);
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
                    preferences: core::mem::take(&mut preferences[vreg]),
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
                !self.target.spill_scratch(interval.class).contains(&reg)
                    && !fixed.get(reg.index() as usize).is_some_and(|ranges| {
                        ranges[ranges.partition_point(|r| r.end < interval.start)..]
                            .iter()
                            .any(|r| {
                                r.value != Some(interval.reg)
                                    && r.start <= interval.end
                                    && interval.start <= r.end
                            })
                    })
            };
            let mut candidates = self
                .target
                .desc()
                .allocatable_regs_in_class(interval.class)
                .to_vec();
            candidates.sort_by_key(|reg| {
                std::cmp::Reverse(interval.preferences.iter().filter(|r| *r == reg).count())
            });
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
                physical.assigned[index].push(Resident {
                    start,
                    end,
                    value: Reg::new_vreg(vreg.as_u32()),
                    ty: source.vreg_data(Reg::new_vreg(vreg.as_u32())).ty,
                });
            }
        }
        for ranges in &mut physical.assigned {
            ranges.sort_unstable_by_key(|r| r.start);
        }
        let instructions = self.plan(&source, &mut frame, &physical)?;
        let incoming = source
            .entry_bindings()
            .iter()
            .map(|binding| {
                Ok(super::moves::Move {
                    dst: self.location(binding.value)?,
                    src: super::moves::Location::Reg(binding.location.into()),
                    ty: source.vreg_data(binding.value).ty,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let entry = super::moves::MoveResolver::default().resolve(
            self.target,
            &mut frame,
            incoming,
            &[],
        )?;
        let edges = self.plan_edges(&mut source, &mut frame)?;
        Ok(Allocation {
            source,
            entry,
            instructions,
            edges,
            frame,
        })
    }

    fn spill(&mut self, reg: Reg, f: &MachineFunction, frame: &mut StackBatch) -> Result<()> {
        let ty = f.vreg_data(reg).ty;
        let layout = &self.target.desc().data_layout;
        // Stack slots remain abstract until the target chooses their base register.
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
        super::operands::plan(self, f, frame, live)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::TargetInfo;
    use crate::target::x86_64::{
        X86_64TargetMachine,
        inst::{
            REG_AF, REG_CF, REG_OF, REG_PF, REG_RAX, REG_RCX, REG_RDX, REG_SF, REG_ZF, TargetInst,
        },
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
                    live.assigned[index].push(Resident {
                        start: 0,
                        end: 100,
                        value: Reg::new_vreg(1000),
                        ty: Type::I64,
                    });
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
                &[dst, REG_CF, REG_PF, REG_ZF, REG_SF, REG_OF],
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
            assert!(!plan.before.is_empty() || !plan.after.is_empty());
            if mode == 1 {
                assert_eq!(plan.after.len(), 1);
            }
            if mode == 2 {
                assert_eq!((plan.before.len(), plan.after.len()), (2, 1));
            }
            assert_eq!(
                f.inst(id).defs().collect::<Vec<_>>(),
                [dst, REG_CF, REG_PF, REG_ZF, REG_SF, REG_OF, REG_AF]
            );
            assert_eq!(f.inst(id).uses().collect::<Vec<_>>(), [rhs, lhs]);
            f.check_refs().unwrap();
            let physical = Allocation {
                source: f,
                entry: Vec::new(),
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

fn reserve(fixed: &mut Vec<Vec<Reservation>>, reg: Reg, at: u32, value: Option<Reg>) {
    let index = reg.index() as usize;
    fixed.resize_with(fixed.len().max(index + 1), Vec::new);
    fixed[index].push(Reservation {
        start: at,
        end: at,
        value,
    });
}
