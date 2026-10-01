use super::FunctionStage;
use crate::analysis::FunctionAnalysisCtx;
use crate::driver::CodegenOptions;
use crate::target::TargetMachine;
use veloc_profile::Profile;

/// Execution resources owned by the function pipeline. Passes receive a FunctionSession.
pub(crate) struct FunctionPassContext<'a> {
    pub(crate) target: &'a dyn TargetMachine,
    pub(crate) func_sig: &'a veloc_mir::Signature,
    pub(crate) symbols: &'a mut veloc_lir::SymbolTable,
    pub(crate) options: &'a CodegenOptions,
    pub(crate) profile: &'a Profile,
    pub(crate) function_analyses: &'a mut FunctionAnalysisCtx,
    pub(crate) stage: FunctionStage,
    pub(crate) next_run: u32,
}
impl<'a> FunctionPassContext<'a> {
    pub(crate) fn new(
        target: &'a dyn TargetMachine,
        func_sig: &'a veloc_mir::Signature,
        symbols: &'a mut veloc_lir::SymbolTable,
        options: &'a CodegenOptions,
        profile: &'a Profile,
        function_analyses: &'a mut FunctionAnalysisCtx,
    ) -> Self {
        Self {
            target,
            func_sig,
            symbols,
            options,
            profile,
            function_analyses,
            stage: FunctionStage::Generic,
            next_run: 0,
        }
    }
}

/// Module passes have no unrestricted access to function-analysis caches.
pub struct ModulePassContext<'a> {
    pub target: &'a dyn TargetMachine,
    pub options: &'a CodegenOptions,
    pub profile: &'a Profile,
    pub(crate) name: String,
    pub(crate) next_run: u32,
}
impl<'a> ModulePassContext<'a> {
    pub fn new(
        target: &'a dyn TargetMachine,
        options: &'a CodegenOptions,
        profile: &'a Profile,
        name: &str,
    ) -> Self {
        Self {
            target,
            options,
            profile,
            name: name.into(),
            next_run: 0,
        }
    }
}
