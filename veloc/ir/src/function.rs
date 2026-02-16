use super::dfg::DataFlowGraph;
use super::layout::Layout;
use super::module::Linkage;
use super::types::{Block, SigId, StackSlot};
use alloc::string::String;
use cranelift_entity::PrimaryMap;

#[derive(Debug, Clone)]
pub struct StackSlotData {
    pub size: u32,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub signature: SigId,
    pub linkage: Linkage,
    pub dfg: DataFlowGraph,
    pub layout: Layout,
    pub stack_slots: PrimaryMap<StackSlot, StackSlotData>,
    pub entry_block: Option<Block>,
    /// 当前函数的修订版本，用于缓存失效
    revision: u64,
}

impl Function {
    pub fn new(name: String, signature: SigId, linkage: Linkage) -> Self {
        Self {
            name,
            signature,
            linkage,
            dfg: DataFlowGraph::new(),
            layout: Layout::new(),
            stack_slots: PrimaryMap::new(),
            entry_block: None,
            revision: 0,
        }
    }

    pub fn is_defined(&self) -> bool {
        self.entry_block.is_some()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn bump_revision(&mut self) {
        self.revision += 1;
    }
}
