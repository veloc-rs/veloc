//! IR-independent control-flow analyses over dense entity IDs.
//! Instruction semantics and cache invalidation remain with each IR adapter.
use alloc::{vec, vec::Vec};
use cranelift_entity::{EntityRef, SecondaryMap};

#[cfg(test)]
mod tests {
    #[test]
    fn exits_and_natural_backedges() {
        use super::*;
        use veloc_mir::Block;
        let b: Vec<_> = (0..5).map(Block::new).collect();
        let mut cfg = ControlFlowGraph::new(b.iter().copied());
        for (a, z) in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 1), (4, 4)] {
            cfg.add_edge(b[a], b[z]);
        }
        let dom = DominatorTree::compute(&cfg, b[0]);
        let post = PostDominatorTree::compute(&cfg);
        assert!(post.post_dominates(b[3], b[0]));
        assert!(!post.post_dominates(b[1], b[0]));
        assert!(!post.post_dominates(b[3], b[4]));
        assert!(post.post_dominates(b[4], b[4]));
        assert_eq!(LoopInfo::compute(&cfg, &dom).backedges(), &[(b[1], b[1])]);
    }
    #[test]
    fn dominators_match_path_removal_including_cycles_and_unreachable_blocks() {
        use super::*;
        use alloc::collections::BTreeSet as HashSet;
        use veloc_mir::Block;
        let mut seed = 17u64;
        for _ in 0..100 {
            let blocks: Vec<_> = (0..10).map(Block::new).collect();
            let mut cfg = ControlFlowGraph::new(blocks.iter().copied());
            for &a in &blocks {
                for &b in &blocks {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if seed >> 60 < 3 {
                        cfg.add_edge(a, b);
                    }
                }
            }
            let reachable = |removed: Option<Block>| {
                let mut seen = HashSet::new();
                let mut pending = vec![blocks[0]];
                while let Some(b) = pending.pop() {
                    if Some(b) != removed && seen.insert(b) {
                        pending.extend_from_slice(cfg.succs(b));
                    }
                }
                seen
            };
            let all = reachable(None);
            let dom = DominatorTree::compute(&cfg, blocks[0]);
            for &a in &blocks {
                let without = reachable(Some(a));
                for &b in &blocks {
                    assert_eq!(
                        dom.dominates(a, b),
                        a == b || all.contains(&b) && !without.contains(&b)
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ControlFlowGraph<B: EntityRef> {
    blocks: Vec<B>,
    preds: SecondaryMap<B, Vec<B>>,
    succs: SecondaryMap<B, Vec<B>>,
}
impl<B: EntityRef> ControlFlowGraph<B> {
    pub fn new(blocks: impl IntoIterator<Item = B>) -> Self {
        Self {
            blocks: blocks.into_iter().collect(),
            preds: SecondaryMap::new(),
            succs: SecondaryMap::new(),
        }
    }
    pub fn blocks(&self) -> &[B] {
        &self.blocks
    }
    pub fn preds(&self, block: B) -> &[B] {
        self.preds.get(block).map_or(&[], Vec::as_slice)
    }
    pub fn succs(&self, block: B) -> &[B] {
        self.succs.get(block).map_or(&[], Vec::as_slice)
    }
    pub fn add_edge(&mut self, from: B, to: B) {
        if !self.succs[from].contains(&to) {
            self.succs[from].push(to);
            self.preds[to].push(from);
        }
    }
}
impl<B: EntityRef> Default for ControlFlowGraph<B> {
    fn default() -> Self {
        Self::new([])
    }
}

#[derive(Debug, Clone)]
pub struct DominatorTree<B: EntityRef> {
    // DFS intervals of the immediate-dominator tree. Unreachable blocks have
    // no interval and must not accidentally dominate reachable blocks.
    intervals: SecondaryMap<B, Option<(usize, usize)>>,
}

impl<B: EntityRef> DominatorTree<B> {
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
        let mut seen = SecondaryMap::<B, bool>::new();
        let mut postorder = Vec::new();
        let mut pending = vec![(entry, false)];
        while let Some((block, leave)) = pending.pop() {
            if leave {
                postorder.push(block);
            } else if !seen[block] {
                seen[block] = true;
                pending.push((block, true));
                pending.extend(cfg.succs(block).iter().rev().map(|&b| (b, false)));
            }
        }
        postorder.reverse();
        let blocks = postorder;
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
        let mut children = vec![Vec::new(); blocks.len()];
        for i in 1..blocks.len() {
            children[parents[i]].push(i);
        }
        let mut intervals = SecondaryMap::<B, Option<(usize, usize)>>::new();
        let mut pending = vec![(0, false)];
        let mut clock = 0;
        while let Some((i, leave)) = pending.pop() {
            let block = blocks[i];
            if leave {
                intervals[block].as_mut().unwrap().1 = clock;
            } else {
                intervals[block] = Some((clock, 0));
                clock += 1;
                pending.push((i, true));
                pending.extend(children[i].iter().rev().map(|&child| (child, false)));
            }
        }
        DominatorTree { intervals }
    }
}
impl<B: EntityRef> Default for DominatorTree<B> {
    fn default() -> Self {
        Self {
            intervals: SecondaryMap::new(),
        }
    }
}

/// Post-dominance with respect to paths reaching an exit. Blocks in non-exiting
/// regions only post-dominate themselves; this does not prove termination.
#[derive(Debug, Clone)]
pub struct PostDominatorTree<B: EntityRef> {
    tree: DominatorTree<B>,
}
impl<B: EntityRef> PostDominatorTree<B> {
    pub fn compute(cfg: &ControlFlowGraph<B>) -> Self {
        let Some(last) = cfg.blocks.iter().map(|b| b.index()).max() else {
            return Self::default();
        };
        let root = B::new(last.checked_add(1).expect("CFG entity index overflow"));
        let mut reverse = ControlFlowGraph::new(cfg.blocks.iter().copied().chain([root]));
        for &block in &cfg.blocks {
            if cfg.succs(block).is_empty() {
                reverse.add_edge(root, block);
            }
            for &succ in cfg.succs(block) {
                reverse.add_edge(succ, block);
            }
        }
        Self {
            tree: DominatorTree::compute(&reverse, root),
        }
    }
    pub fn post_dominates(&self, a: B, b: B) -> bool {
        self.tree.dominates(a, b)
    }
}
impl<B: EntityRef> Default for PostDominatorTree<B> {
    fn default() -> Self {
        Self {
            tree: DominatorTree::default(),
        }
    }
}

/// Natural-loop backedges. Irreducible cycles need a separate SCC analysis.
#[derive(Debug, Clone)]
pub struct LoopInfo<B: EntityRef> {
    backedges: Vec<(B, B)>,
}
impl<B: EntityRef> LoopInfo<B> {
    pub fn compute(cfg: &ControlFlowGraph<B>, dom: &DominatorTree<B>) -> Self {
        let mut backedges = Vec::new();
        for &block in cfg.blocks() {
            if dom.intervals.get(block).copied().flatten().is_none() {
                continue;
            }
            for &succ in cfg.succs(block) {
                if dom.dominates(succ, block) {
                    backedges.push((block, succ));
                }
            }
        }
        Self { backedges }
    }
    pub fn backedges(&self) -> &[(B, B)] {
        &self.backedges
    }
}
impl<B: EntityRef> Default for LoopInfo<B> {
    fn default() -> Self {
        Self {
            backedges: Vec::new(),
        }
    }
}
