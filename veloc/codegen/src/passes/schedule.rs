//! Bounded local list scheduling with register dependencies and pressure control.
//!
//! Only operations explicitly declared movable by the target enter a region.
//! Memory, traps and control effects remain barriers; this needs no alias guesses.
use crate::analysis::{ChangeSet, FunctionAnalysisCtx, PassEffect, RegSet};
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::{RegClass, ScheduleInfo, TargetDescription, TargetSchedule};
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_lir::{InstId, MachineFunction, Reg};

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;

pub struct SchedulePass;

impl FunctionPass for SchedulePass {
    fn name(&self) -> &'static str {
        "schedule"
    }

    fn run(
        &self,
        f: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> crate::Result<PassEffect> {
        if !ctx.options.optimize {
            return Ok(PassEffect::NONE);
        }
        let changed = schedule(f, ctx.target, ctx.function_analyses);
        if ctx.options.collect_stats {
            ctx.stats.scheduled_regions += changed;
        }
        Ok(if changed == 0 {
            PassEffect::NONE
        } else {
            PassEffect::new(ChangeSet::INST_LAYOUT)
        })
    }
}

pub(crate) fn schedule(
    f: &mut MachineFunction,
    target: &dyn TargetSchedule,
    analyses: &mut FunctionAnalysisCtx,
) -> usize {
    // Bound scheduler work even for pathological generated basic blocks.
    const WINDOW: usize = 256;
    let liveness = analyses.liveness(f, target);
    let mut changed = 0;
    let mut ids = Vec::new();
    let mut output = Vec::new();
    let mut info = Vec::new();
    let mut block = f.blocks().next();
    while let Some(b) = block {
        let next_block = f.layout().next_block(b);
        ids.clear();
        ids.extend(f.block_insts(b));
        let mut live: RegSet = liveness.live_out(b).cloned().unwrap_or_default();
        output.clear();
        output.reserve(ids.len());
        let mut end = ids.len();
        // Walk regions backward so live-out is available without storing a live
        // set for every instruction. The actual list scheduler runs forward.
        while end > 0 {
            let mut start = end;
            info.clear();
            while start > 0 && end - start < WINDOW {
                let id = ids[start - 1];
                if f.try_call_info(id).is_some() || f.inst(id).memory().is_some() {
                    break;
                }
                let veloc_lir::MachineOpcode::Target(opcode) = f.inst(id).opcode() else {
                    break;
                };
                // Eligibility and flag effects are instruction facts, not CPU costs.
                let Some(mut cost) = target.instruction_metadata(opcode).schedule else {
                    break;
                };
                if let Some(latency) = target.schedule_latency(opcode) {
                    cost.latency = latency;
                }
                info.push(cost);
                start -= 1;
            }
            if start == end {
                let id = ids[end - 1];
                before(f, id, &mut live);
                output.push(id);
                end -= 1;
                continue;
            }
            if end - start == 1 {
                let id = ids[start];
                before(f, id, &mut live);
                output.push(id);
                end = start;
                continue;
            }
            info.reverse();
            let order = region(f, &ids[start..end], &info, target.desc(), &live);
            if order != ids[start..end] {
                changed += 1;
            }
            output.extend(order.into_iter().rev());
            for &id in ids[start..end].iter().rev() {
                before(f, id, &mut live);
            }
            end = start;
        }
        output.reverse();
        if output != ids {
            f.editor().reorder_block(b, &output);
        }
        block = next_block;
    }
    changed
}

fn before(f: &MachineFunction, id: InstId, live: &mut RegSet) {
    for reg in f.inst(id).defs().chain(f.inst(id).clobbers()) {
        live.remove(&reg);
    }
    for reg in f.inst(id).uses() {
        live.insert(reg);
    }
}

fn bank(class: RegClass) -> usize {
    match class {
        RegClass::GPR => 0,
        RegClass::FPR => 1,
        RegClass::VR => 2,
        RegClass::PR => 3,
        RegClass::Special => 4,
    }
}

fn region(
    f: &MachineFunction,
    ids: &[InstId],
    info: &[ScheduleInfo],
    target: &TargetDescription,
    live_out: &RegSet,
) -> Vec<InstId> {
    let n = ids.len();
    if n < 2 {
        return ids.to_vec();
    }
    let mut edges = vec![SmallVec::<[usize; 4]>::new(); n];
    let mut indegree = vec![0; n];
    let mut writers = HashMap::new();
    let mut readers: HashMap<Reg, SmallVec<[usize; 4]>> = HashMap::new();
    let mut remaining = HashMap::<Reg, usize>::new();
    let mut uses = Vec::with_capacity(n);
    let mut defs = Vec::with_capacity(n);
    let mut edge = |a: usize, b: usize| {
        if a != b && !edges[a].contains(&b) {
            edges[a].push(b);
            indegree[b] += 1;
        }
    };
    for (i, &id) in ids.iter().enumerate() {
        let mut read: SmallVec<[_; 4]> = f.inst(id).uses().collect();
        read.sort();
        read.dedup();
        let mut write: SmallVec<[_; 2]> = f.inst(id).defs().chain(f.inst(id).clobbers()).collect();
        write.sort();
        write.dedup();
        for &r in &read {
            if let Some(&w) = writers.get(&r) {
                edge(w, i);
            }
            readers.entry(r).or_default().push(i);
            *remaining.entry(r).or_default() += 1;
        }
        for &r in &write {
            if let Some(w) = writers.insert(r, i) {
                edge(w, i);
            }
            for reader in readers.remove(&r).unwrap_or_default() {
                edge(reader, i);
            }
        }
        uses.push(read);
        defs.push(write);
    }
    // No instruction in this region reads flags. Earlier flag writes are dead,
    // but the final write must remain last among flag writers for the next region.
    if let Some(last) = info.iter().rposition(|i| i.writes_flags) {
        for (i, cost) in info[..last].iter().enumerate() {
            if cost.writes_flags {
                edge(i, last);
            }
        }
    }
    let mut height = vec![0u32; n];
    for i in (0..n).rev() {
        height[i] = info[i]
            .latency
            .saturating_add(edges[i].iter().map(|&j| height[j]).max().unwrap_or(0));
    }
    let mut ready: Vec<_> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut available = vec![0u32; n];
    let mut cycle = 0u32;
    let mut out = Vec::with_capacity(n);
    let mut live = live_out.clone();
    for &id in ids.iter().rev() {
        before(f, id, &mut live);
    }
    let mut capacity = [0isize; 5];
    for class in target.registers.reg_classes {
        capacity[bank(class.kind)] = class.allocatable.len() as isize;
    }
    let reg_bank = |r| {
        let data = f.vreg_data(r);
        bank(target.reg_class_for_vreg(&data.ty, data.bank))
    };
    let mut pressure = [0isize; 5];
    for reg in live.iter() {
        if reg.is_vreg() {
            pressure[reg_bank(reg)] += 1;
        }
    }
    while !ready.is_empty() {
        let needed_after = |i: usize, r: &Reg| {
            remaining.get(r).copied().unwrap_or(0) > usize::from(uses[i].contains(r))
                || live_out.contains(r)
        };
        let next_pressure = |i: usize| {
            let mut next = pressure;
            for &r in uses[i]
                .iter()
                .chain(defs[i].iter().filter(|r| !uses[i].contains(r)))
            {
                if r.is_vreg() {
                    next[reg_bank(r)] +=
                        isize::from(needed_after(i, &r)) - isize::from(live.contains(&r));
                }
            }
            next
        };
        let best = (0..ready.len())
            .min_by_key(|&k| {
                let i = ready[k];
                let next = next_pressure(i);
                let excess: isize = next
                    .iter()
                    .zip(capacity)
                    .map(|(&p, c)| (p - c).max(0))
                    .sum();
                let total: isize = next.iter().sum();
                (
                    excess,
                    available[i].saturating_sub(cycle),
                    core::cmp::Reverse(height[i]),
                    total,
                    i,
                )
            })
            .unwrap();
        let i = ready.swap_remove(best);
        pressure = next_pressure(i);
        let changes: SmallVec<[_; 8]> = uses[i]
            .iter()
            .chain(&defs[i])
            .map(|&r| (r, needed_after(i, &r)))
            .collect();
        for (reg, needed) in changes {
            if needed {
                live.insert(reg);
            } else {
                live.remove(&reg);
            }
        }
        cycle = cycle.max(available[i]);
        for r in &uses[i] {
            *remaining.get_mut(r).unwrap() -= 1;
        }
        out.push(ids[i]);
        for &j in &edges[i] {
            available[j] = available[j].max(cycle.saturating_add(info[i].latency));
            indegree[j] -= 1;
            if indegree[j] == 0 {
                ready.push(j);
            }
        }
        cycle = cycle.saturating_add(1);
    }
    assert_eq!(out.len(), n, "scheduler dependency graph must be acyclic");
    out
}
