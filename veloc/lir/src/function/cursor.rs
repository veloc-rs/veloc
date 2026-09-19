use super::{InstId, MachineFunction};
use crate::BlockId;

/// Allocation-free traversal for local edits. Capture the next instruction
/// before editing the returned one. Insertions adjacent to the current
/// instruction are skipped; replacing/removing the current instruction is safe.
///
/// The pending instructions and block order must remain in place. CFG rewrites
/// and edits to arbitrary future instructions need a worklist instead.
pub struct InstCursor {
    block: Option<BlockId>,
    next: Option<InstId>,
    single_block: bool,
}

impl InstCursor {
    pub fn new(function: &MachineFunction) -> Self {
        let block = function.blocks().next();
        Self {
            block,
            next: block.and_then(|b| function.layout().first_inst(b)),
            single_block: false,
        }
    }

    pub fn block(function: &MachineFunction, block: BlockId) -> Self {
        assert!(function.layout().contains_block(block));
        Self {
            block: Some(block),
            next: function.layout().first_inst(block),
            single_block: true,
        }
    }

    pub fn next(&mut self, function: &MachineFunction) -> Option<InstId> {
        while let Some(block) = self.block {
            assert!(
                function.layout().contains_block(block),
                "cursor block was removed"
            );
            if let Some(id) = self.next {
                assert_eq!(
                    function.inst_block(id),
                    Some(block),
                    "cursor continuation was removed or moved"
                );
                self.next = function.layout().next_inst(id);
                return Some(id);
            }
            self.block = if self.single_block {
                None
            } else {
                function.layout().next_block(block)
            };
            self.next = self.block.and_then(|b| function.layout().first_inst(b));
        }
        None
    }
}
