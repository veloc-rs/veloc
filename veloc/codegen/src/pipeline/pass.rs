use crate::error::Result;
use crate::pipeline::{CompiledModule, FunctionPassContext, ModulePassContext, PassEffect};
use veloc_lir::MachineFunction;

pub trait FunctionPass<S> {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        mfunc: &mut MachineFunction<S>,
        ctx: &mut FunctionPassContext<'_, S>,
    ) -> Result<PassEffect>;
}

pub trait StageTransformPass<In, Out> {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        mfunc: MachineFunction<In>,
        ctx: &mut FunctionPassContext<'_, In>,
    ) -> Result<(MachineFunction<Out>, PassEffect)>;
}

pub trait ModuleCodegenPass {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        module: &mut CompiledModule,
        ctx: &mut ModulePassContext<'_>,
    ) -> Result<PassEffect>;
}
