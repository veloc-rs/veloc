use crate::analysis::{FunctionAnalysisCtx, ModuleAnalysisCtx};
use crate::driver::CodegenOptions;
use veloc_profile::Profile;

use crate::target::TargetMachine;

pub struct FunctionPassContext<'a> {
    pub target: &'a dyn TargetMachine,
    pub func_sig: &'a veloc_mir::Signature,
    /// Shared symbol identities, including runtime calls introduced by lowering.
    pub symbols: &'a mut veloc_lir::SymbolTable,
    pub options: &'a CodegenOptions,
    pub profile: &'a Profile,
    pub function_analyses: &'a mut FunctionAnalysisCtx,
    pub module_analyses: &'a mut ModuleAnalysisCtx,
}

impl<'a> FunctionPassContext<'a> {
    pub fn new(
        target: &'a dyn TargetMachine,
        func_sig: &'a veloc_mir::Signature,
        symbols: &'a mut veloc_lir::SymbolTable,
        options: &'a CodegenOptions,
        profile: &'a Profile,
        function_analyses: &'a mut FunctionAnalysisCtx,
        module_analyses: &'a mut ModuleAnalysisCtx,
    ) -> Self {
        Self {
            target,
            func_sig,
            symbols,
            options,
            profile,
            function_analyses,
            module_analyses,
        }
    }
}

pub struct ModulePassContext<'a> {
    pub target: &'a dyn TargetMachine,
    pub options: &'a CodegenOptions,
    pub profile: &'a Profile,
    pub module_analyses: &'a mut ModuleAnalysisCtx,
}

impl<'a> ModulePassContext<'a> {
    pub fn new(
        target: &'a dyn TargetMachine,
        options: &'a CodegenOptions,
        profile: &'a Profile,
        module_analyses: &'a mut ModuleAnalysisCtx,
    ) -> Self {
        Self {
            target,
            options,
            profile,
            module_analyses,
        }
    }
}
