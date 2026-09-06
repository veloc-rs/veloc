use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{TargetOperandLowering, TargetPassConfig};
use veloc_lir::stages::PreIselPrepared;

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

    fn apply_effect(effect: PassEffect, ctx: &mut FunctionPassContext<'_, PreIselPrepared>) {
        if !effect.change_set.is_empty() {
            ctx.function_analyses.apply(effect.change_set);
        }
    }
}

impl<'a> StageTransformPass<PreIselPrepared, PreIselPrepared> for PreIselPass<'a> {
    fn name(&self) -> &'static str {
        "pre-isel"
    }

    fn run(
        &self,
        mut mfunc: veloc_lir::MachineFunction<PreIselPrepared>,
        ctx: &mut FunctionPassContext<'_, PreIselPrepared>,
    ) -> Result<(veloc_lir::MachineFunction<PreIselPrepared>, PassEffect)> {
        for pass in self.pass_config.pre_isel_passes() {
            let effect = pass.run(&mut mfunc, ctx)?;
            Self::apply_effect(effect, ctx);
        }

        let pass =
            crate::passes::constraints::PreSelectOperandConstraintPass::new(self.operand_lowering);
        let effect = FunctionPass::run(&pass, &mut mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        // Effects are applied incrementally above so inner passes see fresh analyses.
        Ok((mfunc, PassEffect::NONE))
    }
}
