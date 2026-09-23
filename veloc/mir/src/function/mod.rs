//! Functions, block layout and structural editing.

use crate::dfg::DataFlowGraph;
use crate::{Block, Linkage, SigId, Value};
use alloc::string::String;

mod cfg;
mod dominance;
mod edit;
mod layout;
pub use cfg::ControlFlowGraph;
pub use dominance::Dominators;
pub use edit::{EdgeRef, FuncEditor, InstCursor};
pub use layout::Layout;

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: String,
    pub signature: SigId,
    pub linkage: Linkage,
}

/// Borrowed declaration and optional definition; never owns or duplicates either.
#[derive(Debug, Clone, Copy)]
pub struct FunctionRef<'a> {
    pub decl: &'a FuncDecl,
    pub body: Option<&'a FuncBody>,
}

impl<'a> FunctionRef<'a> {
    pub fn is_defined(&self) -> bool {
        self.body.is_some()
    }
    pub fn entry_block(&self) -> Option<Block> {
        self.body.map(FuncBody::entry_block)
    }
}

/// Function-local IR. Declarations and signatures belong to the module.
#[derive(Debug, Clone)]
pub struct FuncBody {
    dfg: DataFlowGraph,
    layout: Layout,
    cfg: ControlFlowGraph,
    entry_block: Block,
}

impl Default for FuncBody {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl FuncBody {
    /// Create a body with a placed entry block and its signature parameters.
    pub fn new(params: &[crate::Type]) -> Self {
        Self::with_entry(params, Block(0))
    }

    /// The text parser preserves the entry block number from the input.
    pub(crate) fn with_entry(params: &[crate::Type], entry_block: Block) -> Self {
        let mut dfg = DataFlowGraph::new();
        while dfg.blocks.len() <= entry_block.0 as usize {
            dfg.create_block();
        }
        for &ty in params {
            dfg.append_block_param(entry_block, ty);
        }
        let mut layout = Layout::new();
        layout.append_block(entry_block);
        Self {
            dfg,
            layout,
            cfg: ControlFlowGraph::default(),
            entry_block,
        }
    }
    pub fn edit(&mut self) -> FuncEditor<'_> {
        FuncEditor::new(self)
    }
    pub fn dfg(&self) -> &DataFlowGraph {
        &self.dfg
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn cfg(&self) -> &ControlFlowGraph {
        &self.cfg
    }
    pub fn entry_block(&self) -> Block {
        self.entry_block
    }
    pub fn params(&self) -> &[Value] {
        self.dfg.block_params(self.entry_block)
    }
}
