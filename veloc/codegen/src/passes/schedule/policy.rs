//! Choose a scheduling objective once per region, outside the ready-node loop.
use super::{
    NodeId, Region,
    graph::DependencyGraph,
    pressure::{PressureScore, PressureTracker},
    timing::ScheduleTiming,
};
use crate::analysis::RegSet;
use crate::target::TargetSchedule;
use cranelift_entity::EntityRef;

veloc_policy::feature_set!(ScheduleFeatures {
    nodes,
    edges,
    roots,
    critical_path,
    latency_sum,
    latency_max,
    occupancy_sum,
    live_out,
    reads,
    writes,
    memory_ops,
    fanout_max,
    loop_depth,
    edges_per_node,
    root_fraction,
    memory_fraction,
    latency_parallelism,
    issue_cycles_to_path,
    resource_cycles_to_path,
    source_peak_pressure,
});

pub(super) const SCHEMA: veloc_policy::DecisionSchema = veloc_policy::DecisionSchema {
    name: "schedule",
    version: 3,
    features: ScheduleFeatures::NAMES,
    actions: &["balanced", "pressure", "latency", "source_order", "fanout"],
    scope: "function",
};

#[derive(Clone, Copy)]
pub(super) struct Context<'a> {
    pub policy: Option<&'a veloc_policy::Session<'a>>,
    pub loop_depth: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Strategy {
    Balanced,
    Pressure,
    Latency,
    SourceOrder,
    Fanout,
}

impl Context<'_> {
    pub fn choose(
        self,
        region: Region<'_>,
        graph: &DependencyGraph,
        timing: &ScheduleTiming,
        live_out: &RegSet,
        target: &dyn TargetSchedule,
    ) -> Strategy {
        let Some(policy) = self.policy.filter(|p| p.wants_features()) else {
            return Strategy::Balanced;
        };
        let mut features = ScheduleFeatures {
            nodes: region.insts.len() as f32,
            live_out: live_out.len() as f32,
            loop_depth: self.loop_depth as f32,
            ..Default::default()
        };
        let model = target.schedule_model();
        let mut resource_work = vec![0.0_f32; model.resources.len()];
        let mut pressure = PressureTracker::new(region, target.desc(), live_out);
        features.source_peak_pressure = pressure.max_relative_pressure();
        for node in region.nodes() {
            let inst = region.inst(node);
            let cost = timing.costs[node];
            features.edges += graph.edges[node].len() as f32;
            features.roots += f32::from(graph.indegree[node] == 0);
            features.critical_path = features.critical_path.max(timing.height[node] as f32);
            features.latency_sum += cost.latency as f32;
            features.latency_max = features.latency_max.max(cost.latency as f32);
            features.occupancy_sum += cost.occupancy as f32;
            features.reads += inst.register_access().reads().count() as f32;
            features.writes += inst.register_access().writes().count() as f32;
            features.memory_ops += f32::from(inst.mem_flags().is_some());
            features.fanout_max = features.fanout_max.max(graph.edges[node].len() as f32);
            resource_work[cost.resource.index()] += cost.occupancy as f32;
            pressure.advance(node);
            features.source_peak_pressure = features
                .source_peak_pressure
                .max(pressure.max_relative_pressure());
        }
        let nodes = features.nodes.max(1.0);
        let path = features.critical_path.max(1.0);
        features.edges_per_node = features.edges / nodes;
        features.root_fraction = features.roots / nodes;
        features.memory_fraction = features.memory_ops / nodes;
        features.latency_parallelism = features.latency_sum / path;
        features.issue_cycles_to_path = nodes / model.issue_width as f32 / path;
        // A resource lower bound under the selected CPU model, not a runtime
        // prediction. Pool indices and names never enter the feature vector.
        features.resource_cycles_to_path = resource_work
            .iter()
            .zip(model.resources)
            .map(|(work, resource)| work / resource.units as f32 / path)
            .fold(0.0, f32::max);
        match policy.choose(&features.values(), 0) {
            0 => Strategy::Balanced,
            1 => Strategy::Pressure,
            2 => Strategy::Latency,
            3 => Strategy::SourceOrder,
            4 => Strategy::Fanout,
            _ => unreachable!(),
        }
    }
}

impl Strategy {
    pub fn priority(
        self,
        pressure: PressureScore,
        height: u32,
        fanout: usize,
        node: NodeId,
    ) -> [i64; 4] {
        let excess = pressure.excess as i64;
        let total = pressure.total as i64;
        let height = -(height as i64);
        let node = node.index() as i64;
        match self {
            Self::Balanced => [excess, height, total, node],
            Self::Pressure => [excess, total, height, node],
            Self::Latency => [height, excess, total, node],
            Self::Fanout => [excess, -(fanout as i64), height, node],
            Self::SourceOrder => unreachable!("original order is already legal"),
        }
    }
}
