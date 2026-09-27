//! IR-independent control-flow storage and dominators over dense entity IDs.
//! Snapshots are invalidated by CFG changes, not ordinary instruction edits.
use alloc::{vec, vec::Vec};
use cranelift_entity::{EntityRef, SecondaryMap};

#[derive(Debug, Clone)]
struct Edges<B> {
    present: bool,
    preds: Vec<B>,
    succs: Vec<B>,
}

impl<B> Default for Edges<B> {
    fn default() -> Self {
        Self {
            present: false,
            preds: Vec::new(),
            succs: Vec::new(),
        }
    }
}

/// Unique adjacency, not successor occurrences or execution layout. Blocks
/// retain registration order, including explicitly registered isolated blocks.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph<B: EntityRef> {
    blocks: Vec<B>,
    edges: SecondaryMap<B, Edges<B>>,
}

impl<B: EntityRef> Default for ControlFlowGraph<B> {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            edges: SecondaryMap::new(),
        }
    }
}

impl<B: EntityRef> ControlFlowGraph<B> {
    pub fn new(blocks: impl IntoIterator<Item = B>) -> Self {
        let mut cfg = Self::default();
        for block in blocks {
            cfg.add_block(block);
        }
        cfg
    }

    pub fn blocks(&self) -> &[B] {
        &self.blocks
    }

    pub fn add_block(&mut self, block: B) {
        if !self.edges[block].present {
            self.edges[block].present = true;
            self.blocks.push(block);
        }
    }

    pub fn preds(&self, block: B) -> &[B] {
        &self.edges[block].preds
    }
    pub fn succs(&self, block: B) -> &[B] {
        &self.edges[block].succs
    }

    /// Register both endpoints, including forward references during IR building.
    pub fn add_edge(&mut self, from: B, to: B) {
        self.add_block(from);
        self.add_block(to);
        if !self.edges[from].succs.contains(&to) {
            self.edges[from].succs.push(to);
            self.edges[to].preds.push(from);
        }
    }

    /// Replace outgoing edges and synchronize reverse links. Duplicate targets
    /// collapse to one edge; first-occurrence order is retained.
    pub fn set_successors(&mut self, block: B, successors: &[B]) {
        self.add_block(block);
        if self.succs(block) == successors {
            return;
        }
        let mut old = core::mem::take(&mut self.edges[block].succs);
        for &succ in &old {
            if !successors.contains(&succ) {
                self.edges[succ].preds.retain(|&pred| pred != block);
            }
        }
        for &succ in successors {
            self.add_block(succ);
            if !old.contains(&succ) && !self.edges[succ].preds.contains(&block) {
                self.edges[succ].preds.push(block);
            }
        }
        old.clear();
        for &succ in successors {
            if !old.contains(&succ) {
                old.push(succ);
            }
        }
        self.edges[block].succs = old;
    }

    pub fn compute_post_order(&self, entry: B) -> Vec<B> {
        let mut seen = SecondaryMap::<B, bool>::new();
        let mut order = Vec::new();
        let mut pending = vec![(entry, false)];
        while let Some((block, leave)) = pending.pop() {
            if leave {
                order.push(block);
            } else if !seen[block] {
                seen[block] = true;
                pending.push((block, true));
                pending.extend(self.succs(block).iter().rev().map(|&b| (b, false)));
            }
        }
        order
    }

    pub fn compute_rpo(&self, entry: B) -> Vec<B> {
        let mut order = self.compute_post_order(entry);
        order.reverse();
        order
    }
}

#[derive(Debug, Clone)]
pub struct DominatorTree<B: EntityRef> {
    // DFS intervals of the immediate-dominator tree. Unreachable blocks have
    // no interval and must not accidentally dominate reachable blocks.
    intervals: SecondaryMap<B, Option<(usize, usize)>>,
    parents: SecondaryMap<B, Option<B>>,
    first: SecondaryMap<B, Option<B>>,
    next: SecondaryMap<B, Option<B>>,
}

impl<B: EntityRef> DominatorTree<B> {
    /// Entry and unreachable blocks have no immediate dominator.
    pub fn immediate_dominator(&self, block: B) -> Option<B> {
        self.parents[block]
    }

    pub fn is_reachable(&self, block: B) -> bool {
        self.intervals[block].is_some()
    }

    pub fn children(&self, block: B) -> impl Iterator<Item = B> + '_ {
        core::iter::successors(self.first[block], |&child| self.next[child])
    }

    /// Dominance is reflexive, including unreachable blocks. Distinct
    /// unreachable blocks never dominate each other.
    pub fn dominates(&self, a: B, b: B) -> bool {
        if a == b {
            return true;
        }
        match (
            self.intervals.get(a).copied().flatten(),
            self.intervals.get(b).copied().flatten(),
        ) {
            (Some((start, end)), Some((point, _))) => start <= point && point < end,
            _ => false,
        }
    }
    pub fn compute(cfg: &ControlFlowGraph<B>, entry: B) -> Self {
        let blocks = cfg.compute_rpo(entry);
        let mut index = SecondaryMap::<B, Option<usize>>::new();
        for (i, &block) in blocks.iter().enumerate() {
            index[block] = Some(i);
        }
        // Iterative immediate dominators in reverse postorder. Intersect by
        // climbing parent indices, avoiding quadratic sets of dominators.
        let mut parents = vec![usize::MAX; blocks.len()];
        parents[0] = 0;
        loop {
            let mut changed = false;
            for (i, &block) in blocks.iter().enumerate().skip(1) {
                let mut preds = cfg
                    .preds(block)
                    .iter()
                    .filter_map(|&b| index[b])
                    .filter(|&p| parents[p] != usize::MAX);
                let Some(mut parent) = preds.next() else {
                    continue;
                };
                for mut pred in preds {
                    while parent != pred {
                        while parent > pred {
                            parent = parents[parent];
                        }
                        while pred > parent {
                            pred = parents[pred];
                        }
                    }
                }
                if parents[i] != parent {
                    parents[i] = parent;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut idom = SecondaryMap::new();
        let mut first = SecondaryMap::new();
        let mut next = SecondaryMap::new();
        // Prepend in reverse RPO to retain deterministic child order.
        for i in (1..blocks.len()).rev() {
            let block = blocks[i];
            let parent = blocks[parents[i]];
            idom[block] = Some(parent);
            next[block] = first[parent];
            first[parent] = Some(block);
        }
        let mut intervals = SecondaryMap::<B, Option<(usize, usize)>>::new();
        let mut pending = vec![(entry, false)];
        let mut clock = 0;
        while let Some((block, leave)) = pending.pop() {
            if leave {
                intervals[block].as_mut().unwrap().1 = clock;
            } else {
                intervals[block] = Some((clock, 0));
                clock += 1;
                pending.push((block, true));
                let mut child = first[block];
                while let Some(block) = child {
                    pending.push((block, false));
                    child = next[block];
                }
            }
        }
        DominatorTree {
            intervals,
            parents: idom,
            first,
            next,
        }
    }
}
impl<B: EntityRef> Default for DominatorTree<B> {
    fn default() -> Self {
        Self {
            intervals: SecondaryMap::new(),
            parents: SecondaryMap::new(),
            first: SecondaryMap::new(),
            next: SecondaryMap::new(),
        }
    }
}
