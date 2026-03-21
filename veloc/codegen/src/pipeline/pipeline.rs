use crate::error::Result;
use crate::mir::MachineFunction;
use crate::pipeline::{
    ChangeSet, CompiledModule, FunctionPass, FunctionPassContext, ModuleCodegenPass,
    ModulePassContext, PassEffect,
};
use alloc::boxed::Box;
use alloc::vec::Vec;

pub struct StagePassPipeline<S> {
    passes: Vec<Box<dyn FunctionPass<S>>>,
}

impl<S> StagePassPipeline<S> {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass<P: FunctionPass<S> + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    pub fn add_boxed_pass(&mut self, pass: Box<dyn FunctionPass<S>>) {
        self.passes.push(pass);
    }

    pub fn run(
        &self,
        mfunc: &mut MachineFunction<S>,
        ctx: &mut FunctionPassContext<'_, S>,
    ) -> Result<PassEffect> {
        let mut combined = PassEffect::NONE;
        for pass in &self.passes {
            let effect = pass.run(mfunc, ctx)?;
            if !effect.change_set.is_empty() {
                ctx.function_analyses.apply(effect.change_set);
                combined.change_set |= effect.change_set;
            }
        }
        Ok(combined)
    }
}

impl<S> Default for StagePassPipeline<S> {
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
