//! Functions, block layout and structural editing.

use crate::dfg::DataFlowGraph;
use crate::{Block, Linkage, SigId, Value};
use alloc::string::String;

mod edit;
mod layout;
pub type ControlFlowGraph = veloc_collections::graph::ControlFlowGraph<Block>;
pub use edit::{EdgeRef, FuncEditor, InstCursor};
pub use layout::{InstOrder, Layout};

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
    params: alloc::vec::Vec<Value>,
}

impl Default for FuncBody {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl FuncBody {
    /// Create function inputs and a placed entry block without block parameters.
    pub fn new(params: &[crate::Type]) -> Self {
        Self::with_entry(params, Block(0))
    }

    /// The text parser preserves the entry block number from the input.
    pub(crate) fn with_entry(params: &[crate::Type], entry_block: Block) -> Self {
        let mut dfg = DataFlowGraph::new();
        while dfg.blocks.len() <= entry_block.0 as usize {
            dfg.create_block();
        }
        let mut layout = Layout::new();
        layout.append_block(entry_block);
        let mut body = Self {
            dfg,
            layout,
            cfg: ControlFlowGraph::new([entry_block]),
            entry_block,
            params: alloc::vec::Vec::new(),
        };
        for &ty in params {
            body.edit().append_function_param(ty);
        }
        body
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
    /// Function entry. Valid MIR has no control-flow edges into this block;
    /// it has no block parameters. Function inputs are stored separately.
    pub fn entry_block(&self) -> Block {
        self.entry_block
    }
    pub fn params(&self) -> &[Value] {
        &self.params
    }
}
