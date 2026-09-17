//! Functions, block layout and structural editing.

use crate::dfg::DataFlowGraph;
use crate::{Block, Linkage, SigId, Value};
use alloc::{boxed::Box, string::String};

mod cfg;
mod dominance;
mod edit;
mod layout;
pub use cfg::ControlFlowGraph;
pub use dominance::Dominators;
pub use edit::{EdgeRef, FuncEditor};
pub use layout::Layout;

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub signature: SigId,
    pub linkage: Linkage,
    body: Option<Box<FuncBody>>,
}

/// Function-local IR with an allocated entry, even while its contents are built.
#[derive(Debug, Clone)]
pub struct FuncBody {
    dfg: DataFlowGraph,
    layout: Layout,
    cfg: ControlFlowGraph,
    entry_block: Block,
}

impl Function {
    pub fn new(name: String, signature: SigId, linkage: Linkage) -> Self {
        Self {
            name,
            signature,
            linkage,
            body: None,
        }
    }

    pub fn body(&self) -> Option<&FuncBody> {
        self.body.as_deref()
    }

    pub(crate) fn define_body(&mut self) -> &mut FuncBody {
        self.body.get_or_insert_with(|| {
            let mut dfg = DataFlowGraph::new();
            let entry_block = dfg.create_block();
            Box::new(FuncBody {
                dfg,
                layout: Layout::new(),
                cfg: ControlFlowGraph::default(),
                entry_block,
            })
        })
    }

    pub fn body_mut(&mut self) -> Option<&mut FuncBody> {
        self.body.as_deref_mut()
    }

    pub fn entry_block(&self) -> Option<Block> {
        self.body.as_ref().map(|body| body.entry_block)
    }

    pub fn cfg(&self) -> &ControlFlowGraph {
        &self.body().expect("function has no body").cfg
    }

    pub fn is_defined(&self) -> bool {
        self.body.is_some()
    }

    pub fn dfg(&self) -> &DataFlowGraph {
        &self.body().expect("function has no body").dfg
    }

    pub fn layout(&self) -> &Layout {
        &self.body().expect("function has no body").layout
    }

    pub fn edit(&mut self) -> FuncEditor<'_> {
        self.body_mut().expect("cannot edit a declaration").edit()
    }

    /// 获取函数的参数列表（入口块的定义参数）
    pub fn params(&self) -> &[Value] {
        if let Some(entry) = self.entry_block() {
            &self.dfg().blocks[entry].params
        } else {
            &[]
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl FuncBody {
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
}
