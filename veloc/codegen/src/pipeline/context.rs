use crate::driver::{CodegenOptions, CodegenStats};
use crate::pipeline::{FunctionAnalysisCtx, ModuleAnalysisCtx};
use crate::target::arch::TargetMachine;
use core::marker::PhantomData;

pub struct FunctionPassContext<'a, S> {
    pub target: &'a dyn TargetMachine,
    pub func_sig: &'a veloc_mir::Signature,
    pub options: &'a CodegenOptions,
    pub stats: &'a mut CodegenStats,
    pub function_analyses: &'a mut FunctionAnalysisCtx,
    pub module_analyses: &'a mut ModuleAnalysisCtx,
    _stage: PhantomData<S>,
}

impl<'a, S> FunctionPassContext<'a, S> {
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
            _stage: PhantomData,
        }
    }

    pub fn into_stage<T>(self) -> FunctionPassContext<'a, T> {
        FunctionPassContext {
            target: self.target,
            func_sig: self.func_sig,
            options: self.options,
            stats: self.stats,
            function_analyses: self.function_analyses,
            module_analyses: self.module_analyses,
            _stage: PhantomData,
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
