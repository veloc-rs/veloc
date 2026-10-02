//! Ordering constraints are independent of scheduling priorities and CPU resources.
use super::{NodeId, Region};
use crate::analysis::RegSet;
use cranelift_entity::PrimaryMap;
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_lir::Reg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DependencyKind {
    Data(Reg),
    Anti(Reg),
    Output(Reg),
    /// Preserve observable memory and trap order; register operands carry data latency.
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Dependency {
    pub successor: NodeId,
    pub kind: DependencyKind,
}

/// Node indices refer to the region's original order; every edge points forward.
pub(super) struct DependencyGraph {
    pub edges: PrimaryMap<NodeId, SmallVec<[Dependency; 4]>>,
    pub indegree: PrimaryMap<NodeId, usize>,
}

impl DependencyGraph {
    pub fn build(region: Region<'_>, live_out: &RegSet) -> Self {
        let mut graph = Self {
            edges: region.nodes().map(|_| SmallVec::new()).collect(),
            indegree: region.nodes().map(|_| 0).collect(),
        };
        let mut definitions = HashMap::<Reg, NodeId>::new();
        let mut resources = HashMap::<Reg, ResourceAccess>::new();
        let mut last_memory = None;
        for node in region.nodes() {
            // Until alias analysis proves independence, keep all accesses in
            // source order. Pure computations may still fill load latency gaps.
            if region.inst(node).mem_flags().is_some() {
                if let Some(previous) = last_memory {
                    graph.edge(previous, node, DependencyKind::Memory);
                }
                last_memory = Some(node);
            }
            let access = region.inst(node).register_access();
            for value in access.reads() {
                if let Some(&producer) = definitions.get(&value) {
                    graph.edge(producer, node, DependencyKind::Data(value));
                }
                if value.is_preg() {
                    resources
                        .entry(value)
                        .or_default()
                        .read(&mut graph, value, node);
                }
            }
            for value in access.writes() {
                if value.is_vreg() {
                    definitions.insert(value, node);
                }
                if value.is_preg() {
                    resources
                        .entry(value)
                        .or_default()
                        .write(&mut graph, value, node);
                }
            }
        }
        // A live-out observes the last write just like a read beyond the region.
        for (unit, mut resource) in resources {
            if live_out.contains(&unit) {
                resource.observe_write(&mut graph, unit);
            }
        }
        graph
    }

    fn edge(&mut self, from: NodeId, to: NodeId, kind: DependencyKind) {
        if from == to {
            return;
        }
        debug_assert!(from < to, "dependency must follow the original order");
        let dependency = Dependency {
            successor: to,
            kind,
        };
        // Keep each register dependency; only identical reasons are redundant.
        if !self.edges[from].contains(&dependency) {
            self.edges[from].push(dependency);
            self.indegree[to] += 1;
        }
    }

    pub fn preserves_dependencies(&self, order: &[NodeId]) -> bool {
        if order.len() != self.indegree.len() {
            return false;
        }
        let mut positions: PrimaryMap<NodeId, Option<usize>> =
            self.edges.keys().map(|_| None).collect();
        for (pos, &node) in order.iter().enumerate() {
            let Some(entry) = positions.get_mut(node) else {
                return false;
            };
            if entry.replace(pos).is_some() {
                return false;
            }
        }
        self.edges.iter().all(|(from, edges)| {
            edges
                .iter()
                .all(|e| positions[from] < positions[e.successor])
        })
    }
}

/// Streaming dependencies for one physical unit. Unobserved writes may reorder
/// among themselves, but must stay between the surrounding observed versions.
#[derive(Default)]
struct ResourceAccess {
    last_write: Option<NodeId>,
    /// Readers of the last observed version, possibly the region's live-in.
    readers: Vec<NodeId>,
    /// Writes since that version, including last_write until it is observed.
    pending_writes: Vec<NodeId>,
}

impl ResourceAccess {
    fn read(&mut self, graph: &mut DependencyGraph, unit: Reg, reader: NodeId) {
        self.observe_write(graph, unit);
        if let Some(writer) = self.last_write {
            graph.edge(writer, reader, DependencyKind::Data(unit));
        }
        self.readers.push(reader);
    }

    fn write(&mut self, graph: &mut DependencyGraph, unit: Reg, writer: NodeId) {
        for &reader in &self.readers {
            graph.edge(reader, writer, DependencyKind::Anti(unit));
        }
        self.last_write = Some(writer);
        self.pending_writes.push(writer);
    }

    /// The first observation protects this write from preceding dead writes.
    /// Later reads of the same version only need their data dependency.
    fn observe_write(&mut self, graph: &mut DependencyGraph, unit: Reg) {
        let Some(writer) = self.pending_writes.pop() else {
            return;
        };
        for previous in self.pending_writes.drain(..) {
            graph.edge(previous, writer, DependencyKind::Output(unit));
        }
        // Previous readers already constrain all writes up to this definition.
        self.readers.clear();
    }
}
