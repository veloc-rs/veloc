//! IR-independent control-flow analyses over dense entity IDs.
//! Instruction semantics and cache invalidation remain with each IR adapter.
use alloc::vec::Vec;
use cranelift_entity::{EntityRef, SecondaryMap};

#[cfg(test)]
mod tests {
    use alloc::vec;
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

pub use veloc_collections::graph::{ControlFlowGraph, DominatorTree};

/// Post-dominance with respect to paths reaching an exit. Blocks in non-exiting
/// regions only post-dominate themselves; this does not prove termination.
#[derive(Debug, Clone)]
pub struct PostDominatorTree<B: EntityRef> {
    tree: DominatorTree<B>,
}
impl<B: EntityRef> PostDominatorTree<B> {
    pub fn compute(cfg: &ControlFlowGraph<B>) -> Self {
        let Some(last) = cfg.blocks().iter().map(|b| b.index()).max() else {
            return Self::default();
        };
        let root = B::new(last.checked_add(1).expect("CFG entity index overflow"));
        let mut reverse = ControlFlowGraph::new(cfg.blocks().iter().copied().chain([root]));
        for &block in cfg.blocks() {
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
    depth: SecondaryMap<B, u32>,
}
impl<B: EntityRef> LoopInfo<B> {
    pub fn compute(cfg: &ControlFlowGraph<B>, dom: &DominatorTree<B>) -> Self {
        let mut backedges = Vec::new();
        for &block in cfg.blocks() {
            if !dom.is_reachable(block) {
                continue;
            }
            for &succ in cfg.succs(block) {
                if dom.dominates(succ, block) {
                    backedges.push((block, succ));
                }
            }
        }
        // Merge latches with the same header before counting nesting. Walking
        // predecessors stops at the header, which dominates every latch.
        let mut depth = SecondaryMap::new();
        let mut headers = Vec::new();
        for &(_, header) in &backedges {
            if headers.contains(&header) {
                continue;
            }
            headers.push(header);
            let mut members = SecondaryMap::<B, bool>::new();
            members[header] = true;
            let mut pending: Vec<_> = backedges
                .iter()
                .filter_map(|&(latch, h)| (h == header).then_some(latch))
                .collect();
            while let Some(block) = pending.pop() {
                if members[block] {
                    continue;
                }
                members[block] = true;
                pending.extend_from_slice(cfg.preds(block));
            }
            for &block in cfg.blocks() {
                if members[block] {
                    depth[block] += 1;
                }
            }
        }
        Self { backedges, depth }
    }
    pub fn backedges(&self) -> &[(B, B)] {
        &self.backedges
    }
    pub fn depth(&self, block: B) -> u32 {
        self.depth[block]
    }
}
impl<B: EntityRef> Default for LoopInfo<B> {
    fn default() -> Self {
        Self {
            backedges: Vec::new(),
            depth: SecondaryMap::new(),
        }
    }
}
