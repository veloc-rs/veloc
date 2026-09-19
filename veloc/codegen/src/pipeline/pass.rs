use crate::analysis::PassEffect;
use crate::error::Result;
use crate::pipeline::{CompiledModule, FunctionPassContext, ModulePassContext};
use veloc_lir::MachineFunction;

pub trait FunctionPass {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        mfunc: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect>;
}

pub trait ModuleCodegenPass {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        module: &mut CompiledModule,
        ctx: &mut ModulePassContext<'_>,
    ) -> Result<PassEffect>;
}
