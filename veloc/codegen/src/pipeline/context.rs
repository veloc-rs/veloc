use crate::analysis::{FunctionAnalysisCtx, ModuleAnalysisCtx};
use crate::driver::{CodegenOptions, CodegenStats};

use crate::target::TargetMachine;

pub struct FunctionPassContext<'a> {
    pub target: &'a dyn TargetMachine,
    pub func_sig: &'a veloc_mir::Signature,
    pub options: &'a CodegenOptions,
    pub stats: &'a mut CodegenStats,
    pub function_analyses: &'a mut FunctionAnalysisCtx,
    pub module_analyses: &'a mut ModuleAnalysisCtx,
}

impl<'a> FunctionPassContext<'a> {
    pub fn new(
        target: &'a dyn TargetMachine,
        func_sig: &'a veloc_mir::Signature,
        options: &'a CodegenOptions,
        stats: &'a mut CodegenStats,
        function_analyses: &'a mut FunctionAnalysisCtx,
        module_analyses: &'a mut ModuleAnalysisCtx,
    ) -> Self {
        Self {
            target,
            func_sig,
            options,
            stats,
            function_analyses,
            module_analyses,
        }
    }
}

pub struct ModulePassContext<'a> {
    pub target: &'a dyn TargetMachine,
    pub options: &'a CodegenOptions,
    pub stats: &'a mut CodegenStats,
    pub module_analyses: &'a mut ModuleAnalysisCtx,
}

impl<'a> ModulePassContext<'a> {
    pub fn new(
        target: &'a dyn TargetMachine,
        options: &'a CodegenOptions,
        stats: &'a mut CodegenStats,
        module_analyses: &'a mut ModuleAnalysisCtx,
    ) -> Self {
        Self {
            target,
            options,
            stats,
            module_analyses,
        }
    }
}
