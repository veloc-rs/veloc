//! Local list scheduling with register dependencies and pressure control.
//!
//! Only operations explicitly declared movable by the target enter a region.
//! Memory, traps and control effects remain barriers; this needs no alias guesses.
use crate::analysis::{LivenessInfo, RegSet};
use crate::passes::state::StateValues;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::{ScheduleInfo, TargetSchedule};
mod graph;
mod machine;
mod pressure;
mod timing;
use graph::DependencyGraph;
use machine::MachineState;
use pressure::PressureTracker;
use std::vec::Vec;
use timing::ScheduleTiming;
use veloc_lir::{BlockId, InstId, InstRef, MachineFunction, MachineOpcode};

/// All region analyses borrow the same immutable IR until a plan is committed.
#[derive(Clone, Copy)]
struct Region<'a> {
    function: &'a MachineFunction,
    insts: &'a [InstId],
    states: &'a StateValues,
}

impl<'a> Region<'a> {
    fn inst(self, index: usize) -> InstRef<'a> {
        self.function.inst(self.insts[index])
    }

    fn schedule_info(self, index: usize, target: &dyn TargetSchedule) -> ScheduleInfo {
        schedule_info(self.function, target, self.insts[index])
            .expect("scheduling region contains a boundary")
    }
}

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
        let states = StateValues::collect(cx.function(), target)?;
        let plan = cx.with_liveness(|function, liveness| {
            plan_schedule(function, target, liveness, &states, self.verify)
        });
        cx.profile
            .count("scheduled_regions", plan.changed_regions as u64);
        for (block, order) in plan.orders {
            cx.reorder_block(block, &order);
        }
        Ok(())
    }
}

struct SchedulePlan {
    orders: Vec<(BlockId, Vec<InstId>)>,
    changed_regions: usize,
}

struct BlockSchedule {
    order: Vec<InstId>,
    changed_regions: usize,
}

fn plan_schedule(
    f: &MachineFunction,
    target: &dyn TargetSchedule,
    liveness: &LivenessInfo,
    states: &StateValues,
    verify: bool,
) -> SchedulePlan {
    let mut orders = Vec::new();
    let mut changed_regions = 0;
    for block in f.blocks() {
        if let Some(result) = schedule_block(f, block, target, liveness, states, verify) {
            changed_regions += result.changed_regions;
            orders.push((block, result.order));
        }
    }
    SchedulePlan {
        orders,
        changed_regions,
    }
}

fn schedule_block(
    f: &MachineFunction,
    block: BlockId,
    target: &dyn TargetSchedule,
    liveness: &LivenessInfo,
    states: &StateValues,
    verify: bool,
) -> Option<BlockSchedule> {
    let original: Vec<_> = f.block_insts(block).collect();
    let mut order = original.clone();
    let mut live = liveness.live_out(block).cloned().unwrap_or_default();
    let mut changed_regions = 0;
    let mut end = original.len();

    // At each iteration, live describes the original sequence at end. Walk
    // backward to obtain each region's live-out, but keep results in forward order.
    while end > 0 {
        let start = find_region_start(f, target, &original, end);
        // An empty region means the preceding instruction is a boundary.
        // Consume it as a singleton; no scheduling is needed.
        let start = if start == end { end - 1 } else { start };
        let region = &original[start..end];
        if region.len() > 1 {
            let scheduled = schedule_region(f, region, target, &live, states, verify);
            if scheduled != region {
                order[start..end].copy_from_slice(&scheduled);
                changed_regions += 1;
            }
        }

        // Legal reordering preserves the region's boundary liveness.
        for &id in region.iter().rev() {
            before(f.inst(id), &mut live);
        }
        end = start;
    }

    (changed_regions != 0).then_some(BlockSchedule {
        order,
        changed_regions,
    })
}

fn find_region_start(
    f: &MachineFunction,
    target: &dyn TargetSchedule,
    ids: &[InstId],
    end: usize,
) -> usize {
    let mut start = end;
    while start > 0 && schedule_info(f, target, ids[start - 1]).is_some() {
        start -= 1;
    }
    start
}

fn schedule_info(
    f: &MachineFunction,
    target: &dyn TargetSchedule,
    id: InstId,
) -> Option<ScheduleInfo> {
    let inst = f.inst(id);
    if f.try_call_info(id).is_some() || inst.memory().is_some() {
        return None;
    }
    // Generic operations, including call-frame boundaries, remain barriers.
    let MachineOpcode::Target(opcode) = inst.opcode() else {
        return None;
    };
    // Eligibility is an instruction fact, independent of CPU costs.
    target.instruction_metadata(opcode).schedule
}

fn before(inst: InstRef<'_>, live: &mut RegSet) {
    let access = inst.register_access();
    for reg in access.writes() {
        live.remove(&reg);
    }
    for reg in access.reads() {
        live.insert(reg);
    }
}

fn schedule_region(
    f: &MachineFunction,
    ids: &[InstId],
    target: &dyn TargetSchedule,
    live_out: &RegSet,
    states: &StateValues,
    verify: bool,
) -> Vec<InstId> {
    let region = Region {
        function: f,
        insts: ids,
        states,
    };
    let graph = DependencyGraph::build(region, live_out);
    let timing = ScheduleTiming::new(region, &graph, target);
    let mut pressure = PressureTracker::new(region, target.desc(), live_out);
    let mut machine = MachineState::new(target.schedule_model());
    let mut indegree = graph.indegree.clone();
    let mut unlocked: Vec<_> = (0..ids.len()).filter(|&i| indegree[i] == 0).collect();
    let mut available = vec![0u32; ids.len()];
    let mut order = Vec::with_capacity(ids.len());
    while !unlocked.is_empty() {
        // Only issue candidates whose dependencies and resources are ready.
        let earliest = unlocked
            .iter()
            .map(|&i| machine.earliest(available[i], timing.costs[i]))
            .min()
            .unwrap();
        machine.advance_to(earliest);
        let best = (0..unlocked.len())
            .filter(|&k| {
                let i = unlocked[k];
                machine.earliest(available[i], timing.costs[i]) == machine.cycle
            })
            .min_by_key(|&k| {
                let i = unlocked[k];
                let (excess, total) = pressure.score(i);
                (excess, core::cmp::Reverse(timing.height[i]), total, i)
            })
            .unwrap();
        let i = unlocked.swap_remove(best);
        pressure.advance(i);
        order.push(i);
        for edge in &graph.edges[i] {
            let j = edge.successor;
            available[j] = available[j].max(machine.cycle.saturating_add(timing.latency(i, edge)));
            indegree[j] -= 1;
            if indegree[j] == 0 {
                unlocked.push(j);
            }
        }
        machine.issue(timing.costs[i]);
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
