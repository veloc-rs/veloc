//! Immediate dominators for the entry-reachable CFG. This is a snapshot: rebuild
//! it after changing control flow. Instruction order is checked separately.

use super::Layout;
use crate::Block;
use alloc::vec::Vec;

pub struct Dominators {
    parents: Vec<Option<Block>>,
    enter: Vec<usize>,
    exit: Vec<usize>,
}

impl Dominators {
    /// Requires valid CFG adjacency and an entry present in the layout.
    pub fn compute(layout: &Layout, entry: Block) -> Self {
        let count = layout.blocks.len();
        let order = layout.compute_rpo(entry);
        let mut rank = vec![usize::MAX; count];
        for (index, block) in order.iter().enumerate() {
            rank[block.0 as usize] = index;
        }
        let mut parents = vec![None; count];
        parents[entry.0 as usize] = Some(entry);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in order.iter().skip(1) {
                let mut preds = layout.blocks[block]
                    .preds
                    .iter()
                    .copied()
                    .filter(|p| parents[p.0 as usize].is_some());
                let Some(mut parent) = preds.next() else {
                    continue;
                };
                for mut pred in preds {
                    while parent != pred {
                        while rank[parent.0 as usize] > rank[pred.0 as usize] {
                            parent = parents[parent.0 as usize].unwrap();
                        }
                        while rank[pred.0 as usize] > rank[parent.0 as usize] {
                            pred = parents[pred.0 as usize].unwrap();
                        }
                    }
                }
                if parents[block.0 as usize] != Some(parent) {
                    parents[block.0 as usize] = Some(parent);
                    changed = true;
                }
            }
        }

        // Number the dominator tree for constant-time dominance queries, using
        // flat child lists rather than allocating a vector for every block.
        let mut first = vec![None; count];
        let mut next = vec![None; count];
        for &block in order.iter().skip(1) {
            let parent = parents[block.0 as usize].unwrap();
            next[block.0 as usize] = first[parent.0 as usize];
            first[parent.0 as usize] = Some(block);
        }
        let mut enter = vec![usize::MAX; count];
        let mut exit = vec![usize::MAX; count];
        let mut stack = vec![(entry, false)];
        let mut clock = 0;
        while let Some((block, visited)) = stack.pop() {
            let index = block.0 as usize;
            if visited {
                exit[index] = clock;
                continue;
            }
            enter[index] = clock;
            clock += 1;
            stack.push((block, true));
            let mut child = first[index];
            while let Some(block) = child {
                stack.push((block, false));
                child = next[block.0 as usize];
            }
        }
        parents[entry.0 as usize] = None;
        Self {
            parents,
            enter,
            exit,
        }
    }

    /// None for the entry and for unreachable blocks.
    pub fn immediate_dominator(&self, block: Block) -> Option<Block> {
        self.parents[block.0 as usize]
    }

    pub fn is_reachable(&self, block: Block) -> bool {
        self.enter[block.0 as usize] != usize::MAX
    }

    /// Unreachable blocks are outside this tree and dominate no block.
    pub fn dominates(&self, definition: Block, use_block: Block) -> bool {
        self.is_reachable(definition)
            && self.is_reachable(use_block)
            && self.enter[definition.0 as usize] <= self.enter[use_block.0 as usize]
            && self.enter[use_block.0 as usize] < self.exit[definition.0 as usize]
    }
}
