//! Local list scheduling with register dependencies and pressure control.
//!
//! Only operations explicitly declared movable by the target enter a region.
//! Memory accesses retain source order. Volatile, unknown and control effects
//! remain barriers; no alias independence is assumed.
use crate::analysis::{LivenessInfo, RegSet};
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::{ScheduleClass, ScheduleClassId, TargetSchedule};
mod graph;
mod machine;
mod pressure;
mod state;
mod timing;
use graph::DependencyGraph;
use machine::MachineState;
use pressure::PressureTracker;
use state::StateOrder;
use std::vec::Vec;
use timing::ScheduleTiming;
use veloc_lir::{BlockId, InstId, InstRef, MachineFunction, MachineOpcode};

/// All region analyses borrow the same immutable IR until a plan is committed.
#[derive(Clone, Copy)]
struct Region<'a> {
    function: &'a MachineFunction,
    insts: &'a [InstId],
    has_states: bool,
}

impl<'a> Region<'a> {
    fn inst(self, index: usize) -> InstRef<'a> {
        self.function.inst(self.insts[index])
    }

    fn schedule_class(self, index: usize, target: &dyn TargetSchedule) -> ScheduleClassId {
        movable_class(self.function, target, self.insts[index])
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
        let plan = cx.with_liveness(|function, liveness| {
            plan_schedule(function, target, liveness, self.verify)
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
    verify: bool,
) -> SchedulePlan {
    let mut orders = Vec::new();
    let mut changed_regions = 0;
    let has_states = f.vregs().values().any(|data| data.state_unit().is_some());
    for block in f.blocks() {
        if let Some(result) = schedule_block(f, block, target, liveness, has_states, verify) {
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
    has_states: bool,
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
            let scheduled = schedule_region(f, region, target, &live, has_states, verify);
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
    while start > 0 && movable_class(f, target, ids[start - 1]).is_some() {
        start -= 1;
    }
    start
}

fn movable_class(
    f: &MachineFunction,
    target: &dyn TargetSchedule,
    id: InstId,
) -> Option<ScheduleClassId> {
    let inst = f.inst(id);
    // Generic operations, including call-frame boundaries, remain barriers.
    let MachineOpcode::Target(opcode) = inst.opcode() else {
        return None;
    };
    let metadata = target.instruction_metadata(opcode);
    if !metadata.movable {
        return None;
    }
    if inst.mem_flags().is_some_and(|flags| flags.is_volatile()) {
        return None;
    }
    match metadata.schedule_class {
        ScheduleClass::Modeled(class) => Some(class),
        ScheduleClass::Pseudo => None,
    }
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
    has_states: bool,
    verify: bool,
) -> Vec<InstId> {
    let region = Region {
        function: f,
        insts: ids,
        has_states,
    };
    let graph = DependencyGraph::build(region, live_out, &RegSet::default());
    let mut selected = schedule_order(region, target, live_out, &graph, None, verify)
        .expect("scheduler dependency graph must be acyclic");
    // Keep the existing scheduling policy as the fallback. Symbolic lifetimes
    // offer an additional candidate, never a restriction on the accepted IR.
    if has_states {
        let mut states = StateOrder::new(region, live_out);
        let graph = DependencyGraph::build(region, live_out, &states.flexible);
        if let Some(candidate) =
            schedule_order(region, target, live_out, &graph, Some(&mut states), verify)
        {
            if states.preserves_exit() && candidate.cost <= selected.cost {
                selected = candidate;
            }
        }
    }
    selected.order.into_iter().map(|i| ids[i]).collect()
}

struct ScheduledOrder {
    order: Vec<usize>,
    cost: ScheduleScore,
}

/// Compare candidates using the same priorities as list scheduling. These are
/// model estimates; they do not promise an execution-time improvement.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ScheduleScore {
    peak_excess: isize,
    finish: u32,
    peak_pressure: isize,
}

fn schedule_order(
    region: Region<'_>,
    target: &dyn TargetSchedule,
    live_out: &RegSet,
    graph: &DependencyGraph,
    mut states: Option<&mut StateOrder<'_>>,
    verify: bool,
) -> Option<ScheduledOrder> {
    let timing = ScheduleTiming::new(region, graph, target);
    let mut pressure = PressureTracker::new(region, target.desc(), live_out);
    let mut machine = MachineState::new(target.schedule_model());
    let mut indegree = graph.indegree.clone();
    let mut unlocked: Vec<_> = (0..region.insts.len())
        .filter(|&i| indegree[i] == 0)
        .collect();
    let mut available = vec![0u32; region.insts.len()];
    let mut order = Vec::with_capacity(region.insts.len());
    let (mut peak_excess, mut finish, mut peak_pressure) = (0, 0, 0);
    while !unlocked.is_empty() {
        // Only issue candidates whose dependencies and resources are ready.
        let earliest = unlocked
            .iter()
            .filter(|&&i| states.as_ref().is_none_or(|s| s.ready(i)))
            .map(|&i| machine.earliest(available[i], timing.costs[i]))
            .min()?;
        machine.advance_to(earliest);
        let best = (0..unlocked.len())
            .filter(|&k| {
                let i = unlocked[k];
                states.as_ref().is_none_or(|s| s.ready(i))
                    && machine.earliest(available[i], timing.costs[i]) == machine.cycle
            })
            .min_by_key(|&k| {
                let i = unlocked[k];
                let (excess, total) = pressure.score(i);
                (excess, core::cmp::Reverse(timing.height[i]), total, i)
            })
            .unwrap();
        let i = unlocked.swap_remove(best);
        if region.has_states {
            let (excess, total) = pressure.score(i);
            peak_excess = peak_excess.max(excess);
            peak_pressure = peak_pressure.max(total);
            finish = finish.max(machine.cycle.saturating_add(timing.costs[i].latency));
        }
        pressure.advance(i);
        if let Some(states) = states.as_mut() {
            states.advance(i);
        }
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
        region.insts.len(),
        "scheduler dependency graph must be acyclic"
    );
    if verify {
        assert!(
            graph.preserves_dependencies(&order),
            "invalid scheduling permutation"
        );
    }
    Some(ScheduledOrder {
        order,
        cost: ScheduleScore {
            peak_excess,
            finish,
            peak_pressure,
        },
    })
}
