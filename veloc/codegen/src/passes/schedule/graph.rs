//! Ordering constraints are independent of scheduling priorities and CPU resources.
use super::Region;
use crate::analysis::RegSet;
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_lir::Reg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DependencyKind {
    Data(Reg),
    Anti(Reg),
    Output(Reg),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Dependency {
    pub successor: usize,
    pub kind: DependencyKind,
}

/// Node indices refer to the region's original order; every edge points forward.
pub(super) struct DependencyGraph {
    pub edges: Vec<SmallVec<[Dependency; 4]>>,
    pub indegree: Vec<usize>,
}

impl DependencyGraph {
    pub fn build(region: Region<'_>, live_out: &RegSet) -> Self {
        let mut graph = Self {
            edges: vec![SmallVec::new(); region.insts.len()],
            indegree: vec![0; region.insts.len()],
        };
        let mut definitions = HashMap::<Reg, usize>::new();
        let mut resources = HashMap::<Reg, ResourceAccess>::new();
        for i in 0..region.insts.len() {
            let access = region.inst(i).register_access();
            for value in access.reads() {
                if let Some(&producer) = definitions.get(&value) {
                    graph.edge(producer, i, DependencyKind::Data(value));
                }
                let unit = region.states.physical(value);
                if unit.is_preg() {
                    let resource = resources.entry(unit).or_default();
                    let writer = resource.writes.last().copied();
                    resource.readers.entry(writer).or_default().push(i);
                }
            }
            for value in access.writes() {
                if value.is_vreg() {
                    definitions.insert(value, i);
                }
                let unit = region.states.physical(value);
                if unit.is_preg() {
                    let writes = &mut resources.entry(unit).or_default().writes;
                    if writes.last() != Some(&i) {
                        writes.push(i);
                    }
                }
            }
        }
        let mut live_units = RegSet::default();
        for value in live_out.iter() {
            live_units.insert(region.states.physical(value));
        }
        for (unit, resource) in resources {
            graph.protect_resource(unit, resource, live_units.contains(&unit));
        }
        graph
    }

    /// Preserve each observed version, without serializing unobserved writes.
    /// The DAG keeps interfering writes on their original side of each read;
    /// it does not attempt the disjunctive scheduling of fixed live intervals.
    fn protect_resource(&mut self, unit: Reg, resource: ResourceAccess, live_out: bool) {
        let mut readers = resource.readers.get(&None).cloned().unwrap_or_default();
        let mut pending = Vec::new();
        let last = resource.writes.last().copied();
        for writer in resource.writes {
            for &reader in &readers {
                self.edge(reader, writer, DependencyKind::Anti(unit));
            }
            let uses = resource.readers.get(&Some(writer));
            if uses.is_some() || (live_out && Some(writer) == last) {
                for previous in pending.drain(..) {
                    self.edge(previous, writer, DependencyKind::Output(unit));
                }
                readers = uses.cloned().unwrap_or_default();
                for &reader in &readers {
                    self.edge(writer, reader, DependencyKind::Data(unit));
                }
            } else {
                pending.push(writer);
            }
        }
    }

    fn edge(&mut self, from: usize, to: usize, kind: DependencyKind) {
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

    pub fn preserves_dependencies(&self, order: &[usize]) -> bool {
        if order.len() != self.indegree.len() {
            return false;
        }
        let mut positions = vec![None; order.len()];
        for (pos, &node) in order.iter().enumerate() {
            let Some(entry) = positions.get_mut(node) else {
                return false;
            };
            if entry.replace(pos).is_some() {
                return false;
            }
        }
        self.edges.iter().enumerate().all(|(from, edges)| {
            edges
                .iter()
                .all(|e| positions[from] < positions[e.successor])
        })
    }
}

/// Instruction indices only; the IR remains the authority for operands.
#[derive(Default)]
struct ResourceAccess {
    writes: Vec<usize>,
    readers: HashMap<Option<usize>, Vec<usize>>,
}
