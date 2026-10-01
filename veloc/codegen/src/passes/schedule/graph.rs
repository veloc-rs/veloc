//! Ordering constraints are independent of scheduling priorities and CPU resources.
use crate::target::{ScheduleCost, ScheduleInfo, ScheduleModel};
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_lir::{InstId, MachineFunction, Reg};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DependencyKind {
    Data,
    Anti,
    Output,
    Flags,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Dependency {
    pub successor: usize,
    pub kind: DependencyKind,
    pub latency: u32,
}

pub(super) struct Node {
    pub inst: InstId,
    pub uses: SmallVec<[Reg; 4]>,
    pub defs: SmallVec<[Reg; 2]>,
    pub cost: ScheduleCost,
}

pub(super) struct DependencyGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<SmallVec<[Dependency; 4]>>,
    pub indegree: Vec<usize>,
    pub height: Vec<u32>,
}

impl DependencyGraph {
    pub fn build(
        f: &MachineFunction,
        ids: &[InstId],
        info: &[ScheduleInfo],
        model: &ScheduleModel,
    ) -> Self {
        let nodes = ids
            .iter()
            .zip(info)
            .map(|(&inst, info)| {
                let mut uses: SmallVec<[_; 4]> = f.inst(inst).uses().collect();
                uses.sort();
                uses.dedup();
                let mut defs: SmallVec<[_; 2]> =
                    f.inst(inst).defs().chain(f.inst(inst).clobbers()).collect();
                defs.sort();
                defs.dedup();
                Node {
                    inst,
                    uses,
                    defs,
                    cost: model.cost(info.class),
                }
            })
            .collect::<Vec<_>>();
        let mut graph = Self {
            nodes,
            edges: vec![SmallVec::new(); ids.len()],
            indegree: vec![0; ids.len()],
            height: vec![0; ids.len()],
        };
        let mut writers = HashMap::<Reg, usize>::new();
        let mut readers = HashMap::<Reg, SmallVec<[usize; 4]>>::new();
        for i in 0..ids.len() {
            for r in graph.nodes[i].uses.clone() {
                if let Some(&w) = writers.get(&r) {
                    graph.edge(w, i, DependencyKind::Data, graph.nodes[w].cost.latency);
                }
                readers.entry(r).or_default().push(i);
            }
            for r in graph.nodes[i].defs.clone() {
                if let Some(w) = writers.insert(r, i) {
                    graph.edge(w, i, DependencyKind::Output, 0);
                }
                for reader in readers.remove(&r).unwrap_or_default() {
                    graph.edge(reader, i, DependencyKind::Anti, 0);
                }
            }
        }
        // Flag readers are boundaries. Preserve the value leaving this region.
        if let Some(last) = info.iter().rposition(|i| i.writes_flags) {
            for (i, cost) in info[..last].iter().enumerate() {
                if cost.writes_flags {
                    graph.edge(i, last, DependencyKind::Flags, 0);
                }
            }
        }
        for i in (0..ids.len()).rev() {
            graph.height[i] = graph.edges[i]
                .iter()
                .map(|e| e.latency.saturating_add(graph.height[e.successor]))
                .max()
                .unwrap_or(0)
                .max(graph.nodes[i].cost.latency);
        }
        graph
    }

    fn edge(&mut self, from: usize, to: usize, kind: DependencyKind, latency: u32) {
        if from == to {
            return;
        }
        // Retain distinct reasons, but merge duplicate constraints of the same kind.
        if let Some(old) = self.edges[from]
            .iter_mut()
            .find(|e| e.successor == to && e.kind == kind)
        {
            old.latency = old.latency.max(latency);
        } else {
            self.edges[from].push(Dependency {
                successor: to,
                kind,
                latency,
            });
            self.indegree[to] += 1;
        }
    }

    pub fn preserves_dependencies(&self, order: &[usize]) -> bool {
        if order.len() != self.nodes.len() {
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
