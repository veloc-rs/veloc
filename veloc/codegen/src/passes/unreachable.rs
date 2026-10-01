//! Remove blocks outside the entry's reachable CFG before instruction selection.
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use cranelift_entity::SecondaryMap;
use std::vec::Vec;
use veloc_lir::BlockId;

pub struct RemoveUnreachablePass;

impl FunctionPass for RemoveUnreachablePass {
    fn name(&self) -> &'static str {
        "remove-unreachable"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Legal
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let entry = cx.function().entry_block();
        let order = cx.cfg().compute_post_order(entry);
        if order.len() == cx.function().num_blocks() {
            return Ok(());
        }
        let mut reachable = SecondaryMap::<BlockId, bool>::new();
        for block in order {
            reachable[block] = true;
        }
        let dead: Vec<_> = cx.function().blocks().filter(|&b| !reachable[b]).collect();
        for block in dead {
            cx.erase_block(block);
        }
        Ok(())
    }
}
