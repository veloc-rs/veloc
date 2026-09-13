//! Bounded local list scheduling with register dependencies and pressure control.
//!
//! Only operations explicitly declared movable by the target enter a region.
//! Memory, traps and control effects remain barriers; this needs no alias guesses.
use crate::pipeline::FunctionAnalysisCtx;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{RegClass, ScheduleInfo, TargetMachine};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashSet;
use veloc_lir::{InstId, MachineFunction, Reg, stages::PostIselOptimized};

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;

pub struct SchedulePass;

impl FunctionPass<PostIselOptimized> for SchedulePass {
    fn name(&self) -> &'static str {
        "schedule"
    }

    fn run(
        &self,
        f: &mut MachineFunction<PostIselOptimized>,
        ctx: &mut FunctionPassContext<'_, PostIselOptimized>,
    ) -> crate::Result<PassEffect> {
        if !ctx.options.optimize {
            return Ok(PassEffect::NONE);
        }
        let changed = schedule(f, ctx.target, ctx.function_analyses);
        ctx.stats.scheduled_regions += changed;
        Ok(if changed == 0 {
            PassEffect::NONE
        } else {
            PassEffect::new(ChangeSet::BLOCK_LAYOUT)
        })
    }
}

pub(crate) fn schedule<S>(
    f: &mut MachineFunction<S>,
    target: &dyn TargetMachine,
    analyses: &mut FunctionAnalysisCtx,
) -> usize {
    // Bound scheduler work even for pathological generated basic blocks.
    const WINDOW: usize = 256;
    let liveness = analyses.liveness(f, target);
    let mut changed = 0;
    for b in 0..f.num_blocks() {
        let ids = f.block_insts(b).to_vec();
        let mut live = liveness
            .live_out(f.blocks[b].id)
            .cloned()
            .unwrap_or_default();
        let mut output = Vec::with_capacity(ids.len());
        let mut end = ids.len();
        // Walk regions backward so live-out is available without storing a live
        // set for every instruction. The actual list scheduler runs forward.
        while end > 0 {
            let mut start = end;
            let mut info = Vec::new();
            while start > 0 && end - start < WINDOW {
                let id = ids[start - 1];
                if f.inst_extra(id).is_some() {
                    break;
                }
                let Some(cost) = target.schedule_info(&f.inst(id)) else {
                    break;
                };
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
            info.reverse();
            let order = region(f, &ids[start..end], &info, target, &live);
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
        f.blocks[b].insts = output;
    }
    changed
}

fn before<S>(f: &MachineFunction<S>, id: InstId, live: &mut HashSet<Reg>) {
    for reg in f.inst(id).defs() {
        live.remove(&reg);
    }
    live.extend(f.inst(id).uses());
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

fn region<S>(
    f: &MachineFunction<S>,
    ids: &[InstId],
    info: &[ScheduleInfo],
    target: &dyn TargetMachine,
    live_out: &HashSet<Reg>,
) -> Vec<InstId> {
    let n = ids.len();
    if n < 2 {
        return ids.to_vec();
    }
    let mut edges = vec![Vec::new(); n];
    let mut indegree = vec![0; n];
    let mut writers = BTreeMap::new();
    let mut readers: BTreeMap<Reg, Vec<usize>> = BTreeMap::new();
    let mut remaining: BTreeMap<Reg, usize> = BTreeMap::new();
    let mut uses = Vec::with_capacity(n);
    let mut defs = Vec::with_capacity(n);
    let mut edge = |a: usize, b: usize| {
        if a != b && !edges[a].contains(&b) {
            edges[a].push(b);
            indegree[b] += 1;
        }
    };
    for (i, &id) in ids.iter().enumerate() {
        let mut read: Vec<_> = f.inst(id).uses().collect();
        read.sort();
        read.dedup();
        let mut write: Vec<_> = f.inst(id).defs().collect();
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
    for class in target.desc().registers.reg_classes {
        capacity[bank(class.kind)] = class.allocatable.len() as isize;
    }
    let reg_bank = |r| {
        let data = f.vreg_data(r);
        bank(target.desc().reg_class_for_vreg(&data.ty, data.bank))
    };
    let mut pressure = [0isize; 5];
    for &reg in &live {
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
        let changes: Vec<_> = uses[i]
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
