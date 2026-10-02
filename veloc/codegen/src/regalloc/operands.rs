//! Choose locations for individual occurrences, then resolve all transfers at
//! each boundary together. The global allocator supplies preferred value homes.
use super::allocation::{InstAllocation, Transfer};
use super::constraints::constraints;
use super::linear_scan::{RegisterAllocator, RegisterLiveness, Resident};
use super::moves::{Location, Move, MoveResolver, storage};
use crate::target::SpillKind;
use crate::{Error, Result};
use cranelift_entity::SecondaryMap;
use hashbrown::HashMap;
use veloc_lir::{
    InstId, InstRef, MachineFunction, OperandRef, PReg, Placement, Reg, StackBatch, StackSlot, Type,
};

struct Group {
    members: Vec<usize>,
    allowed: Vec<Reg>,
    input: Option<Reg>,
    output: Option<Reg>,
    location: Option<Reg>,
}

fn index(operand: OperandRef, inputs: usize) -> usize {
    match operand {
        OperandRef::Input(i) => i,
        OperandRef::Result(i) => inputs + i,
    }
}

fn groups(
    allocator: &RegisterAllocator<'_>,
    f: &MachineFunction,
    inst: InstRef<'_>,
) -> Result<Vec<Group>> {
    let values: Vec<_> = inst
        .inputs()
        .iter()
        .chain(inst.results())
        .copied()
        .collect();
    let mut roots: Vec<_> = (0..values.len()).collect();
    let mut allowed = vec![None::<Vec<Reg>>; values.len()];
    for (i, &value) in values.iter().enumerate() {
        if value.is_preg() {
            allowed[i] = Some(vec![value]);
        }
    }
    for c in constraints(inst, allocator.target) {
        let i = index(c.operand, inst.inputs().len());
        match c.placement {
            Placement::Reuse(input) => {
                let from = roots[i];
                let to = roots[input];
                for root in &mut roots {
                    if *root == from {
                        *root = to;
                    }
                }
            }
            placement => {
                let regs = match placement {
                    Placement::Fixed(reg) => vec![reg],
                    Placement::Registers(regs) => regs.to_vec(),
                    Placement::Reuse(_) => unreachable!(),
                };
                if let Some(current) = &mut allowed[i] {
                    current.retain(|r| regs.contains(r));
                } else {
                    allowed[i] = Some(regs);
                }
            }
        }
    }
    let mut groups = Vec::new();
    for root in 0..values.len() {
        let members: Vec<_> = roots
            .iter()
            .enumerate()
            .filter_map(|(i, r)| (*r == root).then_some(i))
            .collect();
        if members.is_empty() {
            continue;
        }
        let mut group = Group {
            members,
            allowed: Vec::new(),
            input: None,
            output: None,
            location: None,
        };
        for (n, &i) in group.members.iter().enumerate() {
            let value = values[i];
            let value_regs = allowed[i].clone().unwrap_or_else(|| {
                let data = f.vreg_data(value);
                let class = allocator
                    .target
                    .desc()
                    .reg_class_for_vreg(&data.ty, data.bank());
                allocator
                    .target
                    .desc()
                    .registers
                    .reg_class(class)
                    .unwrap()
                    .members
                    .to_vec()
            });
            if n == 0 {
                group.allowed = value_regs;
            } else {
                group.allowed.retain(|r| value_regs.contains(r));
            }
            let part = if i < inst.inputs().len() {
                &mut group.input
            } else {
                &mut group.output
            };
            if part.is_some_and(|old| old != value) {
                return Err(Error::codegen(
                    "reuse group contains distinct simultaneous values",
                ));
            }
            *part = Some(value);
        }
        if group.allowed.is_empty() {
            return Err(Error::codegen("incompatible operand constraints"));
        }
        groups.push(group);
    }
    // Resolve mandatory locations before choosing flexible ones. Among flexible
    // groups, constrained read/write groups precede ordinary inputs and results.
    groups.sort_by_key(|g| {
        (
            g.allowed.len() != 1,
            !(g.input.is_some() && g.output.is_some()),
            g.input.is_none(),
            g.allowed.len(),
        )
    });
    Ok(groups)
}

fn needs_save(resident: Resident, group: &Group, pos: u32) -> bool {
    resident.end > pos + 1
        && (group.input.is_some_and(|value| value != resident.value)
            || group.output.is_some_and(|value| value != resident.value))
}

pub(super) fn plan(
    allocator: &RegisterAllocator<'_>,
    f: &MachineFunction,
    frame: &mut StackBatch,
    live: &RegisterLiveness,
) -> Result<SecondaryMap<InstId, InstAllocation>> {
    let mut plans = SecondaryMap::new();
    let mut slots = HashMap::<(Reg, Type), StackSlot>::new();
    let mut resolver = MoveResolver::default();
    let mut pos = 2;
    for block in f.blocks() {
        for id in f.block_insts(block) {
            let inst = f.inst(id);
            let mut groups = groups(allocator, f, inst)?;
            // A conditional exit may continue, but its taken path skips restores.
            let restore = !allocator.target.control_flow(&inst).may_leave_block();
            for n in 0..groups.len() {
                let g = &groups[n];
                let available = |reg: Reg| {
                    if g.input
                        .is_some_and(|v| v.is_vreg() && live.reserved(reg, pos, v))
                        || g.output
                            .is_some_and(|v| v.is_vreg() && live.reserved(reg, pos + 1, v))
                    {
                        return false;
                    }
                    if !restore
                        && live
                            .resident(reg, pos)
                            .is_some_and(|r| needs_save(r, g, pos))
                    {
                        return false;
                    }
                    groups[..n].iter().all(|other| {
                        if other.location != Some(reg) {
                            return true;
                        }
                        !(g.output.is_some() && other.output.is_some()
                            || g.input.zip(other.input).is_some_and(|(a, b)| a != b))
                    })
                };
                let candidates = g
                    .allowed
                    .iter()
                    .copied()
                    .filter(|&reg| {
                        // Singleton constraints may name reserved ABI registers.
                        g.allowed.len() == 1
                            || allocator
                                .target
                                .desc()
                                .registers
                                .reg_classes
                                .iter()
                                .any(|c| {
                                    c.allocatable.contains(&reg)
                                        || allocator.target.spill_scratch(c.kind).contains(&reg)
                                })
                    })
                    .filter(|&r| available(r));
                let location = candidates
                    .min_by_key(|&reg| {
                        let preservation = live
                            .resident(reg, pos)
                            .is_some_and(|r| needs_save(r, g, pos));
                        let home_cost = [g.input, g.output]
                            .into_iter()
                            .flatten()
                            .filter(|&value| {
                                allocator.assigned(value).map(Reg::from) != Some(reg)
                                    && value != reg
                            })
                            .count();
                        // Keep unrelated inputs in their existing homes when possible.
                        let displaced = g.input.is_some()
                            && inst.inputs().iter().any(|&value| {
                                Some(value) != g.input
                                    && allocator.assigned(value).map(Reg::from) == Some(reg)
                            });
                        (
                            preservation,
                            home_cost + usize::from(displaced),
                            reg.index(),
                        )
                    })
                    .ok_or_else(|| {
                        Error::codegen(format!("cannot satisfy operand locations at {id:?}"))
                    })?;
                groups[n].location = Some(location);
            }

            let mut plan = InstAllocation::default();
            let mut inputs = vec![None::<PReg>; inst.inputs().len()];
            let mut results = vec![None::<PReg>; inst.results().len()];
            let mut before = Vec::new();
            let mut after = Vec::new();
            let mut saved = Vec::new();
            let mut protected = Vec::new();
            for g in &groups {
                let reg = g.location.unwrap();
                for value in [g.input, g.output].into_iter().flatten() {
                    protected.push((reg, value.as_vreg().map(|_| f.vreg_data(value).ty)));
                }
                if let Some(resident) = live.resident(reg, pos).filter(|r| needs_save(*r, g, pos)) {
                    if !saved.iter().any(|&(old, _, _)| old == reg) {
                        let (size, align) = storage(allocator.target, resident.ty)?;
                        let slot = *slots.entry((reg, resident.ty)).or_insert_with(|| {
                            frame.alloc_object(veloc_lir::StackObject::Local, size, align)
                        });
                        saved.push((reg, slot, resident.ty));
                    }
                }
                for &i in &g.members {
                    let (value, write) = if i < inputs.len() {
                        inputs[i] = reg.as_preg();
                        (inst.inputs()[i], false)
                    } else {
                        results[i - inputs.len()] = reg.as_preg();
                        (inst.results()[i - inputs.len()], true)
                    };
                    if value.is_preg() {
                        continue;
                    }
                    let home = allocator.location(value)?;
                    let local = Location::Reg(reg);
                    let ty = f.vreg_data(value).ty;
                    if write {
                        after.push(Move {
                            dst: home,
                            src: local,
                            ty,
                        });
                    } else {
                        before.push(Move {
                            dst: local,
                            src: home,
                            ty,
                        });
                    }
                }
            }
            plan.locations
                .extend(inputs.into_iter().map(|r| r.expect("input location")));
            plan.results
                .extend(results.into_iter().map(|r| r.expect("result location")));
            for &(reg, slot, ty) in &saved {
                plan.before.push(Transfer::Spill {
                    kind: SpillKind::Store,
                    reg,
                    slot,
                    ty,
                });
            }
            plan.before
                .extend(resolver.resolve(allocator.target, frame, before, &protected)?);
            plan.after
                .extend(resolver.resolve(allocator.target, frame, after, &protected)?);
            for (reg, slot, ty) in saved {
                plan.after.push(Transfer::Spill {
                    kind: SpillKind::Load,
                    reg,
                    slot,
                    ty,
                });
            }
            if !restore && !plan.after.is_empty() {
                return Err(Error::codegen(
                    "transfers after a block exit require an edge location",
                ));
            }
            plans[id] = plan;
            pos += 2;
        }
    }
    Ok(plans)
}
