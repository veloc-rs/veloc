//! Local list scheduling with register dependencies and pressure control.
//!
//! Only operations explicitly declared movable by the target enter a region.
//! Nontrapping reads may reorder; writes retain order with every memory access.
//! Volatile, unknown and control effects remain barriers.
//! Selection has already resolved hardware state to physical register operands.
use crate::analysis::{LivenessInfo, RegSet};
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::{ScheduleClass, ScheduleClassId, TargetSchedule};
use cranelift_entity::{EntityRef, PrimaryMap, entity_impl};
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

/// A node in one scheduling region, numbered in original instruction order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeId(u32);
entity_impl!(NodeId, "node");

/// All region analyses borrow the same immutable IR until a plan is committed.
#[derive(Clone, Copy)]
struct Region<'a> {
    function: &'a MachineFunction,
    insts: &'a [InstId],
}

impl<'a> Region<'a> {
    fn nodes(self) -> impl DoubleEndedIterator<Item = NodeId> + ExactSizeIterator {
        (0..self.insts.len()).map(NodeId::new)
    }

    fn inst_id(self, node: NodeId) -> InstId {
        self.insts[node.index()]
    }

    fn inst(self, node: NodeId) -> InstRef<'a> {
        self.function.inst(self.inst_id(node))
    }

    fn schedule_class(self, node: NodeId, target: &dyn TargetSchedule) -> ScheduleClassId {
        movable_class(self.function, target, self.inst_id(node))
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
    for block in f.blocks() {
        if let Some(result) = schedule_block(f, block, target, liveness, verify) {
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
            let scheduled = schedule_region(f, region, target, &live, verify);
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
    verify: bool,
) -> Vec<InstId> {
    let region = Region {
        function: f,
        insts: ids,
    };
    let graph = DependencyGraph::build(region, live_out, target);
    schedule_order(region, target, live_out, &graph, verify)
        .into_iter()
        .map(|node| region.inst_id(node))
        .collect()
}

fn schedule_order(
    region: Region<'_>,
    target: &dyn TargetSchedule,
    live_out: &RegSet,
    graph: &DependencyGraph,
    verify: bool,
) -> Vec<NodeId> {
    let timing = ScheduleTiming::new(region, graph, target);
    let mut pressure = PressureTracker::new(region, target.desc(), live_out);
    let mut machine = MachineState::new(target.schedule_model());
    let mut indegree = graph.indegree.clone();
    let mut unlocked: Vec<_> = region.nodes().filter(|&node| indegree[node] == 0).collect();
    let mut available: PrimaryMap<NodeId, u32> = region.nodes().map(|_| 0).collect();
    let mut order = Vec::with_capacity(region.insts.len());
    while !unlocked.is_empty() {
        // Only issue candidates whose dependencies and resources are ready.
        let earliest = unlocked
            .iter()
            .map(|&node| machine.earliest(available[node], timing.costs[node]))
            .min()
            .unwrap();
        machine.advance_to(earliest);
        let best_position = (0..unlocked.len())
            .filter(|&position| {
                let node = unlocked[position];
                machine.earliest(available[node], timing.costs[node]) == machine.cycle
            })
            .min_by_key(|&position| {
                let node = unlocked[position];
                let score = pressure.score(node);
                (
                    score.excess,
                    core::cmp::Reverse(timing.height[node]),
                    score.total,
                    node,
                )
            })
            .unwrap();
        let node = unlocked.swap_remove(best_position);
        pressure.advance(node);
        order.push(node);
        for edge in &graph.edges[node] {
            let successor = edge.successor;
            available[successor] =
                available[successor].max(machine.cycle.saturating_add(timing.latency(node, edge)));
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                unlocked.push(successor);
            }
        }
        machine.issue(timing.costs[node]);
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
    order
}
