use super::inst::Inst;
use super::types::{Block, Value};
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;

#[derive(Debug, Clone)]
pub struct BlockData {
    pub params: Vec<Value>,
    pub preds: Vec<Block>,
    pub succs: Vec<Block>,
    pub insts: Vec<Inst>,
    pub is_sealed: bool,
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub blocks: PrimaryMap<Block, BlockData>,
    pub block_order: Vec<Block>,
}

impl Layout {
    pub fn new() -> Self {
        Self {
            blocks: PrimaryMap::new(),
            block_order: Vec::new(),
        }
    }

    pub fn create_block(&mut self) -> Block {
        self.blocks.push(BlockData {
            params: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
            insts: Vec::new(),
            is_sealed: false,
        })
    }

    pub fn append_block(&mut self, block: Block) {
        self.block_order.push(block);
    }

    pub fn append_inst(&mut self, block: Block, inst: Inst) {
        self.blocks[block].insts.push(inst);
    }

    pub fn add_edge(&mut self, from: Block, to: Block) {
        if !self.blocks[from].succs.contains(&to) {
            self.blocks[from].succs.push(to);
        }
        if !self.blocks[to].preds.contains(&from) {
            self.blocks[to].preds.push(from);
        }
    }
}
