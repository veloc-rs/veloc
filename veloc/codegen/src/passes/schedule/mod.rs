//! Bounded local list scheduling with register dependencies and pressure control.
//!
//! Only operations explicitly declared movable by the target enter a region.
//! Memory, traps and control effects remain barriers; this needs no alias guesses.
use crate::analysis::{LivenessInfo, RegSet};
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::{ScheduleInfo, TargetSchedule};
mod graph;
mod machine;
mod pressure;
use graph::DependencyGraph;
use machine::MachineState;
use pressure::PressureTracker;
use std::vec::Vec;
use veloc_lir::{InstId, MachineFunction};

pub struct SchedulePass {
    verify: bool,
}
impl SchedulePass {
    pub fn new(verify: bool) -> Self {
        Self { verify }
    }
}

impl FunctionPass for SchedulePass {
    fn name(&self) -> &'static str {
        "schedule"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let target = cx.target;
        let plan = cx.with_liveness(|function, liveness| {
            plan_schedule(function, target, liveness, self.verify)
        });
        cx.profile.count("scheduled_regions", plan.regions as u64);
        for (block, order) in plan.orders {
            cx.reorder_block(block, &order);
        }
        Ok(())
    }
}

struct SchedulePlan {
    orders: Vec<(veloc_lir::BlockId, Vec<InstId>)>,
    regions: usize,
}
fn plan_schedule(
    f: &MachineFunction,
    target: &dyn TargetSchedule,
    liveness: &LivenessInfo,
    verify: bool,
) -> SchedulePlan {
    const WINDOW: usize = 256;
    let mut orders = Vec::new();
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
                // Generic call-frame boundaries remain hard scheduling barriers
                // until frame lowering chooses their physical implementation.
                let veloc_lir::MachineOpcode::Target(opcode) = f.inst(id).opcode() else {
                    break;
                };
                // Eligibility and flag effects are instruction facts, not CPU costs.
                let Some(cost) = target.instruction_metadata(opcode).schedule else {
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
            if end - start == 1 {
                let id = ids[start];
                before(f, id, &mut live);
                output.push(id);
                end = start;
                continue;
            }
            info.reverse();
            let order = region(f, &ids[start..end], &info, target, &live, verify);
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
            orders.push((b, core::mem::take(&mut output)));
        }
        block = next_block;
    }
    SchedulePlan {
        orders,
        regions: changed,
    }
}

fn before(f: &MachineFunction, id: InstId, live: &mut RegSet) {
    for reg in f.inst(id).defs().chain(f.inst(id).clobbers()) {
        live.remove(&reg);
    }
    for reg in f.inst(id).uses() {
        live.insert(reg);
    }
}

fn region(
    f: &MachineFunction,
    ids: &[InstId],
    info: &[ScheduleInfo],
    target: &dyn TargetSchedule,
    live_out: &RegSet,
    verify: bool,
) -> Vec<InstId> {
    let graph = DependencyGraph::build(f, ids, info, target.schedule_model());
    let mut pressure = PressureTracker::new(f, &graph, target.desc(), live_out);
    let mut machine = MachineState::new(target.schedule_model());
    let mut indegree = graph.indegree.clone();
    let mut unlocked: Vec<_> = (0..ids.len()).filter(|&i| indegree[i] == 0).collect();
    let mut available = vec![0u32; ids.len()];
    let mut order = Vec::with_capacity(ids.len());
    while !unlocked.is_empty() {
        // Only issue candidates whose dependencies and resources are ready.
        let earliest = unlocked
            .iter()
            .map(|&i| machine.earliest(available[i], graph.nodes[i].cost))
            .min()
            .unwrap();
        machine.advance_to(earliest);
        let best = (0..unlocked.len())
            .filter(|&k| {
                let i = unlocked[k];
                machine.earliest(available[i], graph.nodes[i].cost) == machine.cycle
            })
            .min_by_key(|&k| {
                let i = unlocked[k];
                let (excess, total) = pressure.score(&graph.nodes[i]);
                (excess, core::cmp::Reverse(graph.height[i]), total, i)
            })
            .unwrap();
        let i = unlocked.swap_remove(best);
        pressure.advance(&graph.nodes[i]);
        order.push(i);
        for edge in &graph.edges[i] {
            let j = edge.successor;
            available[j] = available[j].max(machine.cycle.saturating_add(edge.latency));
            indegree[j] -= 1;
            if indegree[j] == 0 {
                unlocked.push(j);
            }
        }
        machine.issue(graph.nodes[i].cost);
    }
    assert_eq!(
        order.len(),
        ids.len(),
        "scheduler dependency graph must be acyclic"
    );
    if verify {
        assert!(
            graph.preserves_dependencies(&order),
            "invalid scheduling permutation"
        );
    }
    order.into_iter().map(|i| ids[i]).collect()
}
