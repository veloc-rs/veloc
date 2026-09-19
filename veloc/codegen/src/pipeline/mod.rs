pub mod compiled;
pub mod context;
pub mod pass;

pub use compiled::{CompiledFunction, CompiledModule};
pub use context::{FunctionPassContext, ModulePassContext};
pub use pass::{FunctionPass, ModuleCodegenPass};

use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use alloc::boxed::Box;
use alloc::vec::Vec;
use veloc_lir::MachineFunction;

/// Shared execution for built-in and target passes.
pub(crate) fn run_function_pass(
    pass: &dyn FunctionPass,
    function: &mut MachineFunction,
    ctx: &mut FunctionPassContext<'_>,
) -> crate::Result<PassEffect> {
    #[cfg(feature = "std")]
    let start = ctx.options.collect_stats.then(std::time::Instant::now);
    let effect = pass
        .run(function, ctx)
        .map_err(|e| crate::Error::codegen(alloc::format!("{}: {e}", pass.name())))?;
    #[cfg(feature = "std")]
    if let Some(start) = start {
        *ctx.stats.pass_times.entry(pass.name().into()).or_default() += start.elapsed();
    }
    ctx.function_analyses.apply(effect.change_set);
    dump_after(pass.name(), function, ctx.options);
    Ok(effect)
}

pub(crate) fn dump_after(name: &str, function: &MachineFunction, options: &crate::CodegenOptions) {
    #[cfg(feature = "std")]
    if options.dump_after.iter().any(|p| p == "*" || p == name)
        && options
            .dump_function
            .as_deref()
            .is_none_or(|filter| filter == function.name)
    {
        std::eprintln!(
            "===== LIR after {name}: {} =====\n{}",
            function.name,
            function.format_for_dump()
        );
    }
    #[cfg(not(feature = "std"))]
    let _ = (name, function, options);
}

pub struct FunctionPassPipeline {
    passes: Vec<Box<dyn FunctionPass>>,
}

impl FunctionPassPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass<P: FunctionPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    pub fn add_boxed_pass(&mut self, pass: Box<dyn FunctionPass>) {
        self.passes.push(pass);
    }

    pub fn from_passes(passes: Vec<Box<dyn FunctionPass>>) -> Self {
        Self { passes }
    }

    pub fn run(
        &self,
        mfunc: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let mut combined = PassEffect::NONE;
        for pass in &self.passes {
            let effect = run_function_pass(&**pass, mfunc, ctx)?;
            if !effect.change_set.is_empty() {
                combined.change_set |= effect.change_set;
            }
        }
        Ok(combined)
    }
}

impl Default for FunctionPassPipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ModulePassPipeline {
    passes: Vec<Box<dyn ModuleCodegenPass>>,
}

impl ModulePassPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass<P: ModuleCodegenPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    pub fn add_boxed_pass(&mut self, pass: Box<dyn ModuleCodegenPass>) {
        self.passes.push(pass);
    }

    pub fn run(
        &self,
        module: &mut CompiledModule,
        ctx: &mut ModulePassContext<'_>,
    ) -> Result<PassEffect> {
        let mut combined = PassEffect::new(ChangeSet::NONE);
        for pass in &self.passes {
            let effect = pass.run(module, ctx)?;
            if !effect.change_set.is_empty() {
                ctx.module_analyses.apply(effect.change_set);
                combined.change_set |= effect.change_set;
            }
        }
        Ok(combined)
    }
}

impl Default for ModulePassPipeline {
    fn default() -> Self {
        Self::new()
    }
}
