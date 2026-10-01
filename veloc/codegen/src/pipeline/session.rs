//! A pass can query cached analyses or edit a function, but cannot do both at once.
use crate::analysis::{ChangeSet, FunctionAnalysisCtx};
use crate::pipeline::FunctionPassContext;
use crate::target::TargetMachine;
use core::ops::{Deref, DerefMut};
use veloc_lir::{BlockId, InstId, MachineFunction, SymbolTable};

pub struct FunctionSession<'a> {
    function: &'a mut MachineFunction,
    analyses: &'a mut FunctionAnalysisCtx,
    symbols: &'a mut SymbolTable,
    pub target: &'a dyn TargetMachine,
    pub signature: &'a veloc_mir::Signature,
    pub options: &'a crate::CodegenOptions,
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
            options: ctx.options,
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
    /// Compatibility entry for algorithms that still mutate MachineFunction
    /// directly. First mutable access conservatively invalidates all analyses.
    /// The guard exclusively borrows this session, preventing queries until it
    /// is released, including on early return or unwinding. No rollback implied.
    pub fn edit(&mut self) -> FunctionEdit<'_> {
        FunctionEdit {
            function: self.function,
            analyses: self.analyses,
            symbols: self.symbols,
            invalidated: false,
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
}

pub struct FunctionEdit<'a> {
    function: &'a mut MachineFunction,
    analyses: &'a mut FunctionAnalysisCtx,
    symbols: &'a mut SymbolTable,
    invalidated: bool,
}
impl FunctionEdit<'_> {
    fn invalidate(&mut self) {
        if !self.invalidated {
            self.analyses.apply(ChangeSet::WHOLE_FUNCTION);
            self.invalidated = true;
        }
    }
    /// Function passes may intern identities, but cannot rename or mutate other
    /// symbols in the shared table.
    pub fn intern_function(
        &mut self,
        name: &str,
        linkage: veloc_mir::Linkage,
    ) -> veloc_lir::SymbolId {
        self.invalidate();
        self.symbols.get_or_create_function(name, linkage)
    }
    /// Adapter for the existing legalization implementation. Keep unrestricted
    /// table access internal; extension passes use intern_function instead.
    pub(crate) fn with_symbols(&mut self) -> (&mut MachineFunction, &mut SymbolTable) {
        self.invalidate();
        (self.function, self.symbols)
    }
}
impl Deref for FunctionEdit<'_> {
    type Target = MachineFunction;
    fn deref(&self) -> &MachineFunction {
        self.function
    }
}
impl DerefMut for FunctionEdit<'_> {
    fn deref_mut(&mut self) -> &mut MachineFunction {
        self.invalidate();
        self.function
    }
}
