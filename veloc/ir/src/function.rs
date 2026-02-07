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
        }
    }
}
