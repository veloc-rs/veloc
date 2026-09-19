//! Backward liveness independent of an IR's value/register representation.
use crate::graph::ControlFlowGraph;
use cranelift_entity::{EntityRef, SecondaryMap};

/// Set operations required by the liveness equations. The caller selects the
/// representation (e.g. dense bits, sparse IDs, or separate register banks).
pub trait LiveSet: Clone + Default + PartialEq {
    fn clear(&mut self);
    fn union_with(&mut self, other: &Self);
    fn union_difference(&mut self, values: &Self, removed: &Self);
}

#[derive(Debug, Clone)]
pub struct BlockLiveness<B: EntityRef, S: LiveSet> {
    live_in: SecondaryMap<B, S>,
    live_out: SecondaryMap<B, S>,
}

impl<B: EntityRef, S: LiveSet> Default for BlockLiveness<B, S> {
    fn default() -> Self {
        Self {
            live_in: SecondaryMap::new(),
            live_out: SecondaryMap::new(),
        }
    }
}

impl<B: EntityRef, S: LiveSet> BlockLiveness<B, S> {
    pub fn live_in(&self, block: B) -> Option<&S> {
        self.live_in.get(block)
    }
    pub fn live_out(&self, block: B) -> Option<&S> {
        self.live_out.get(block)
    }

    /// Uses are upward-exposed uses; defs include parameters and clobbers as
    /// appropriate for the IR. Edge operands must be accounted for by the caller.
    pub fn compute(
        cfg: &ControlFlowGraph<B>,
        uses: &SecondaryMap<B, S>,
        defs: &SecondaryMap<B, S>,
    ) -> Self {
        let mut result = Self::default();
        for &block in cfg.blocks() {
            result.live_in[block] = S::default();
            result.live_out[block] = S::default();
        }
        // Swap changed results so scratch buffers retain their allocations.
        let mut out = S::default();
        let mut new_in = S::default();
        loop {
            let mut changed = false;
            for &block in cfg.blocks().iter().rev() {
                out.clear();
                for &succ in cfg.succs(block) {
                    out.union_with(&result.live_in[succ]);
                }
                new_in.clone_from(&uses[block]);
                new_in.union_difference(&out, &defs[block]);
                if result.live_out[block] != out {
                    core::mem::swap(&mut result.live_out[block], &mut out);
                    changed = true;
                }
                if result.live_in[block] != new_in {
                    core::mem::swap(&mut result.live_in[block], &mut new_in);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        result
    }
}
