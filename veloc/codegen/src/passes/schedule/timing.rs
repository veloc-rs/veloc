//! CPU costs and critical-path priorities for a scheduling region.
use super::{
    NodeId, Region,
    graph::{Dependency, DependencyGraph, DependencyKind},
};
use crate::target::{ScheduleCost, TargetSchedule};
use cranelift_entity::PrimaryMap;

pub(super) struct ScheduleTiming {
    pub costs: PrimaryMap<NodeId, ScheduleCost>,
    pub height: PrimaryMap<NodeId, u32>,
}

impl ScheduleTiming {
    pub fn new(region: Region<'_>, graph: &DependencyGraph, target: &dyn TargetSchedule) -> Self {
        let model = target.schedule_model();
        let mut timing = Self {
            costs: region
                .nodes()
                .map(|node| model.cost(region.schedule_class(node, target)))
                .collect(),
            height: region.nodes().map(|_| 0).collect(),
        };
        // Graph nodes are in topological order. Include the node's own result
        // latency even when it has no successor within this region.
        for node in region.nodes().rev() {
            timing.height[node] = graph.edges[node]
                .iter()
                .map(|edge| {
                    timing
                        .latency(node, edge)
                        .saturating_add(timing.height[edge.successor])
                })
                .max()
                .unwrap_or(0)
                .max(timing.costs[node].latency);
        }
        timing
    }

    pub fn latency(&self, producer: NodeId, dependency: &Dependency) -> u32 {
        match dependency.kind {
            // The current CPU model gives all results the same latency.
            DependencyKind::Data(_) => self.costs[producer].latency,
            DependencyKind::Anti(_) | DependencyKind::Output(_) | DependencyKind::Memory => 0,
        }
    }
}
