//! Functions, block layout and structural editing.

use crate::dfg::DataFlowGraph;
use crate::{Block, Linkage, SigId, StackSlot, Value};
use alloc::string::String;
use cranelift_entity::PrimaryMap;

mod dominance;
mod edit;
mod layout;
pub use dominance::Dominators;
pub use edit::{EdgeRef, FunctionEditor};
pub use layout::{BlockData, Layout};

#[derive(Debug, Clone)]
pub struct StackSlotData {
    pub size: u32,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub signature: SigId,
    pub linkage: Linkage,
    pub(crate) dfg: DataFlowGraph,
    pub(crate) layout: Layout,
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

    pub fn is_defined(&self) -> bool {
        self.entry_block.is_some()
    }

    pub fn dfg(&self) -> &DataFlowGraph {
        &self.dfg
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn edit(&mut self) -> FunctionEditor<'_> {
        FunctionEditor::new(self)
    }

    /// 获取函数的参数列表（入口块的定义参数）
    pub fn params(&self) -> &[Value] {
        if let Some(entry) = self.entry_block {
            &self.layout.blocks[entry].params
        } else {
            &[]
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}
