//! Form contiguous traces after allocation has inserted edge-copy blocks.
//! Branch encodings and inversions remain the symbolic emitter's responsibility.
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use cranelift_entity::SecondaryMap;
use std::vec::Vec;
use veloc_lir::BlockId;

pub struct BlockPlacementPass;

impl FunctionPass for BlockPlacementPass {
    fn name(&self) -> &'static str {
        "block-placement"
    }
    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Allocated
    }

    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let original: Vec<_> = cx.function().blocks().collect();
        // An implicit fallthrough is tied to layout. Until it is materialized
        // as an explicit edge, retain the original layout for this function.
        if original.iter().any(|&block| {
            cx.function().block_insts(block).all(|id| {
                cx.target
                    .control_flow(&cx.function().inst(id))
                    .may_continue()
            })
        }) {
            return Ok(());
        }
        let cfg = cx.cfg().clone();
        let loops = cx.loop_info().clone();
        let mut positions = SecondaryMap::<BlockId, usize>::new();
        for (position, &block) in original.iter().enumerate() {
            positions[block] = position;
        }
        let mut placed = SecondaryMap::<BlockId, bool>::new();
        let mut order = Vec::with_capacity(original.len());
        let mut seed = Some(cx.function().entry_block());
        while let Some(mut block) = seed {
            loop {
                placed[block] = true;
                order.push(block);
                let next = cfg
                    .succs(block)
                    .iter()
                    .copied()
                    .filter(|&succ| !placed[succ] && loops.depth(succ) >= loops.depth(block))
                    .max_by_key(|&succ| {
                        (
                            loops.depth(succ),
                            cfg.preds(succ).len() == 1,
                            positions[succ] == positions[block] + 1,
                            core::cmp::Reverse(positions[succ]),
                        )
                    });
                let Some(next) = next else { break };
                block = next;
            }
            seed = original
                .iter()
                .copied()
                .filter(|&b| !placed[b])
                .max_by_key(|&b| (loops.depth(b), core::cmp::Reverse(positions[b])));
        }
        cx.reorder_blocks(&order);
        Ok(())
    }
}
