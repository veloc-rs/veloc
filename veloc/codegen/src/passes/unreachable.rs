//! Remove blocks outside the entry's reachable CFG before instruction selection.
use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};
use cranelift_entity::SecondaryMap;
use std::vec::Vec;
use veloc_lir::{BlockId, MachineFunction};

pub struct RemoveUnreachablePass;

impl FunctionPass for RemoveUnreachablePass {
    fn name(&self) -> &'static str {
        "remove-unreachable"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let cfg = ctx.function_analyses.cfg(mfunc, ctx.target);
        let order = cfg.compute_post_order(mfunc.entry_block());
        if order.len() == mfunc.num_blocks() {
            return Ok(PassEffect::NONE);
        }
        let mut reachable = SecondaryMap::<BlockId, bool>::new();
        for block in order {
            reachable[block] = true;
        }
        let dead: Vec<_> = mfunc.blocks().filter(|&block| !reachable[block]).collect();
        let mut edit = mfunc.editor();
        // No live block enters this set. Erasing the whole set also removes
        // internal edges (including cycles) and outgoing edge argument uses.
        for block in dead {
            edit.erase_block(block);
        }
        Ok(PassEffect::new(
            ChangeSet::CFG
                | ChangeSet::INST_LAYOUT
                | ChangeSet::INST_SEMANTICS
                | ChangeSet::INST_OPERANDS,
        ))
    }
}
