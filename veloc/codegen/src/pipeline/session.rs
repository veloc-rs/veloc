//! A pass can query cached analyses or edit a function, but cannot do both at once.
use crate::analysis::{ChangeSet, FunctionAnalysisCtx};
use crate::pipeline::FunctionPassContext;
use crate::target::TargetMachine;
use core::ops::{Deref, DerefMut};
use veloc_lir::{BlockId, FuncEditor, InstId, MachineFunction, SymbolTable};

pub struct FunctionSession<'a> {
    function: &'a mut MachineFunction,
    analyses: &'a mut FunctionAnalysisCtx,
    symbols: &'a mut SymbolTable,
    pub target: &'a dyn TargetMachine,
    pub signature: &'a veloc_mir::Signature,
    pub profile: &'a veloc_profile::Profile,
}
impl<'a> FunctionSession<'a> {
    pub(crate) fn new(
        function: &'a mut MachineFunction,
        ctx: &'a mut FunctionPassContext<'_>,
    ) -> Self {
        Self {
            function,
            analyses: ctx.function_analyses,
            symbols: ctx.symbols,
            target: ctx.target,
            signature: ctx.func_sig,
            profile: ctx.profile,
        }
    }
    pub fn function(&self) -> &MachineFunction {
        self.function
    }
    pub fn cfg(&mut self) -> &crate::analysis::CfgInfo {
        self.analyses.cfg(self.function, self.target)
    }
    pub fn dominators(&mut self) -> &crate::analysis::DominatorTree {
        self.analyses.dominators(self.function, self.target)
    }
    pub fn post_dominators(&mut self) -> &crate::analysis::PostDominatorTree {
        self.analyses.post_dominators(self.function, self.target)
    }
    pub fn loop_info(&mut self) -> &crate::analysis::LoopInfo {
        self.analyses.loop_info(self.function, self.target)
    }
    pub fn register_pressure(&mut self) -> &crate::analysis::RegisterPressure {
        self.analyses.register_pressure(self.function, self.target)
    }
    pub fn stack_frame_summary(&mut self) -> &crate::analysis::StackFrameSummary {
        self.analyses.stack_frame_summary(self.function)
    }
    pub fn liveness(&mut self) -> &crate::analysis::LivenessInfo {
        self.analyses.liveness(self.function, self.target)
    }
    /// Compute an owned plan from a read-only function and its current analysis.
    pub fn with_liveness<R>(
        &mut self,
        inspect: impl FnOnce(&MachineFunction, &crate::analysis::LivenessInfo) -> R,
    ) -> R {
        inspect(
            self.function,
            self.analyses.liveness(self.function, self.target),
        )
    }
    /// Invalidate before granting write access. The editor exclusively borrows
    /// the session and cannot expose mutable function storage. This also covers
    /// early returns and unwinding; edits are not rolled back.
    pub fn edit(&mut self) -> FunctionEdit<'_> {
        self.analyses.apply(ChangeSet::WHOLE_FUNCTION);
        FunctionEdit {
            editor: self.function.editor(),
            symbols: self.symbols,
        }
    }
    /// A constrained edit can precisely record the only kind of change it permits.
    pub fn reorder_block(&mut self, block: BlockId, order: &[InstId]) {
        if self.function.block_insts(block).eq(order.iter().copied()) {
            return;
        }
        self.analyses.apply(ChangeSet::INST_LAYOUT);
        self.function.editor().reorder_block(block, order);
    }
    pub fn erase_block(&mut self, block: BlockId) {
        self.analyses.apply(
            ChangeSet::CFG
                | ChangeSet::INST_LAYOUT
                | ChangeSet::INST_OPERANDS
                | ChangeSet::INST_SEMANTICS,
        );
        self.function.editor().erase_block(block);
    }
    /// Change physical block order without changing instructions or explicit edges.
    pub fn reorder_blocks(&mut self, order: &[BlockId]) {
        assert_eq!(order.first().copied(), Some(self.function.entry_block()));
        assert_eq!(order.len(), self.function.num_blocks());
        let mut seen = cranelift_entity::SecondaryMap::<BlockId, bool>::new();
        for &block in order {
            assert!(self.function.layout().contains_block(block) && !seen[block]);
            seen[block] = true;
        }
        if self.function.blocks().eq(order.iter().copied()) {
            return;
        }
        self.analyses.apply(ChangeSet::BLOCK_LAYOUT);
        let mut edit = self.function.editor();
        for pair in order.windows(2).rev() {
            edit.move_block_before(pair[0], pair[1]);
        }
    }
}

pub struct FunctionEdit<'a> {
    editor: FuncEditor<'a>,
    symbols: &'a mut SymbolTable,
}
impl FunctionEdit<'_> {
    /// Function passes may intern identities, but cannot rename or mutate other
    /// symbols in the shared table.
    pub fn intern_function(
        &mut self,
        name: &str,
        linkage: veloc_mir::Linkage,
    ) -> veloc_lir::SymbolId {
        self.symbols.get_or_create_function(name, linkage)
    }
    /// Adapter for the existing legalization implementation. Keep unrestricted
    /// table access internal; extension passes use intern_function instead.
    pub(crate) fn with_symbols(&mut self) -> (FuncEditor<'_>, &mut SymbolTable) {
        (self.editor.editor(), self.symbols)
    }
}
impl<'a> Deref for FunctionEdit<'a> {
    type Target = FuncEditor<'a>;
    fn deref(&self) -> &Self::Target {
        &self.editor
    }
}
impl DerefMut for FunctionEdit<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.editor
    }
}
