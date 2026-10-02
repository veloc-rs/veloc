//! CPU costs and critical-path priorities for a scheduling region.
use super::{
    Region,
    graph::{Dependency, DependencyGraph, DependencyKind},
};
use crate::target::{ScheduleCost, TargetSchedule};

pub(super) struct ScheduleTiming {
    pub costs: Vec<ScheduleCost>,
    pub height: Vec<u32>,
}

impl ScheduleTiming {
    pub fn new(region: Region<'_>, graph: &DependencyGraph, target: &dyn TargetSchedule) -> Self {
        let model = target.schedule_model();
        let mut timing = Self {
            costs: (0..region.insts.len())
                .map(|i| model.cost(region.schedule_class(i, target)))
                .collect(),
            height: vec![0; region.insts.len()],
        };
        // Graph nodes are in topological order. Include the node's own result
        // latency even when it has no successor within this region.
        for i in (0..region.insts.len()).rev() {
            timing.height[i] = graph.edges[i]
                .iter()
                .map(|edge| {
                    timing
                        .latency(i, edge)
                        .saturating_add(timing.height[edge.successor])
                })
                .max()
                .unwrap_or(0)
                .max(timing.costs[i].latency);
        }
        timing
    }

    pub fn latency(&self, producer: usize, dependency: &Dependency) -> u32 {
        match dependency.kind {
            // The current CPU model gives all results the same latency.
            DependencyKind::Data(_) => self.costs[producer].latency,
            DependencyKind::Anti(_) | DependencyKind::Output(_) | DependencyKind::Memory => 0,
        }
    }
}
