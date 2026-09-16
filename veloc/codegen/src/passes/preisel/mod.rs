use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{TargetOperandLowering, TargetPassConfig};

pub struct PreIselPass<'a> {
    operand_lowering: &'a dyn TargetOperandLowering,
    pass_config: &'a dyn TargetPassConfig,
}

impl<'a> PreIselPass<'a> {
    pub fn new(
        operand_lowering: &'a dyn TargetOperandLowering,
        pass_config: &'a dyn TargetPassConfig,
    ) -> Self {
        Self {
            operand_lowering,
            pass_config,
        }
    }

    fn apply_effect(effect: PassEffect, ctx: &mut FunctionPassContext<'_>) {
        if !effect.change_set.is_empty() {
            ctx.function_analyses.apply(effect.change_set);
        }
    }
}

impl<'a> FunctionPass for PreIselPass<'a> {
    fn name(&self) -> &'static str {
        "pre-isel"
    }

    fn run(
        &self,
        mfunc: &mut veloc_lir::MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        for pass in self.pass_config.pre_isel_passes() {
            let effect = pass.run(mfunc, ctx)?;
            Self::apply_effect(effect, ctx);
        }

        let pass =
            crate::passes::constraints::PreSelectOperandConstraintPass::new(self.operand_lowering);
        let effect = FunctionPass::run(&pass, mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        // Effects are applied incrementally above so inner passes see fresh analyses.
        Ok(PassEffect::NONE)
    }
}
