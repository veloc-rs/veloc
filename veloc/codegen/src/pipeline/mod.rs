pub mod compiled;
pub mod context;
mod function;
mod instrumentation;
pub mod pass;
mod session;

use crate::{Error, Result};
pub use compiled::{CompiledFunction, CompiledModule, EmissionModule, EmittedFunction};
pub(crate) use context::FunctionPassContext;
pub use context::ModulePassContext;
pub use function::FunctionPipeline;
pub(crate) use instrumentation::dump_after;
pub use pass::{FunctionPass, FunctionStage, ModuleCodegenPass};
pub use session::{FunctionEdit, FunctionSession};
use veloc_lir::MachineFunction;

/// Every function pass, including target extensions, uses this execution path.
pub(crate) fn run_function_pass(
    pass: &dyn FunctionPass,
    function: &mut MachineFunction,
    ctx: &mut FunctionPassContext<'_>,
) -> Result<()> {
    let position = ctx.next_run;
    ctx.next_run += 1;
    let profile = ctx.profile;
    let name = function.name.clone();
    let (mut result, observation) =
        instrumentation::execute(profile, pass.name(), position, &name, || {
            if ctx.stage != pass.input_stage() {
                return Err(Error::codegen(format!(
                    "requires {:?}, received {:?}",
                    pass.input_stage(),
                    ctx.stage
                )));
            }
            pass.run(&mut FunctionSession::new(function, ctx))
        });
    if result.is_ok() {
        if ctx.options.verify {
            result = profile
                .measure("verify-pass", position, || {
                    pass.output_stage().verify(function, ctx.target)
                })
                .map_err(|e| {
                    Error::codegen(format!(
                        "{name}/{}#{position}: invalid {:?} output: {e}",
                        pass.name(),
                        pass.output_stage()
                    ))
                });
        }
        if result.is_ok() {
            ctx.stage = pass.output_stage();
        } else if let Err(error) = &result {
            observation.remark(|| error.to_string());
        }
    }
    // Keep a failed pass's partially modified IR for diagnosis. An error aborts
    // this compilation; neither the runner nor an edit guard performs rollback.
    observation.artifact(|| function.format_for_dump().to_string());
    dump_after(pass.name(), function, ctx.options);
    result
}

pub(crate) struct PassSequence {
    passes: Vec<Box<dyn FunctionPass>>,
}
impl PassSequence {
    pub fn from_passes(passes: Vec<Box<dyn FunctionPass>>) -> Self {
        Self { passes }
    }
    pub fn run(
        &self,
        function: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<()> {
        for pass in &self.passes {
            run_function_pass(&**pass, function, ctx)?;
        }
        Ok(())
    }
}

fn run_module_pass<M: core::fmt::Debug>(
    pass: &dyn ModuleCodegenPass<M>,
    module: &mut M,
    ctx: &mut ModulePassContext<'_>,
) -> Result<()> {
    let position = ctx.next_run;
    ctx.next_run += 1;
    let name = ctx.name.clone();
    let (result, observation) =
        instrumentation::execute(ctx.profile, pass.name(), position, &name, || {
            pass.run(module, ctx)
        });
    observation.artifact(|| format!("{module:#?}"));
    result
}

/// The module representation is part of the pass type, so a pre-emission pass
/// cannot accidentally be registered in the post-emission pipeline.
pub struct ModulePassPipeline<M> {
    passes: Vec<Box<dyn ModuleCodegenPass<M>>>,
}
impl<M> Default for ModulePassPipeline<M> {
    fn default() -> Self {
        Self { passes: Vec::new() }
    }
}
impl<M: core::fmt::Debug> ModulePassPipeline<M> {
    pub fn from_passes(passes: Vec<Box<dyn ModuleCodegenPass<M>>>) -> Self {
        Self { passes }
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_pass<P: ModuleCodegenPass<M> + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }
    pub fn add_boxed_pass(&mut self, pass: Box<dyn ModuleCodegenPass<M>>) {
        self.passes.push(pass);
    }
    pub fn run(&self, module: &mut M, ctx: &mut ModulePassContext<'_>) -> Result<()> {
        for pass in &self.passes {
            run_module_pass(&**pass, module, ctx)?;
        }
        Ok(())
    }
}
