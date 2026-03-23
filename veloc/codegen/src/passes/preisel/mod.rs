use crate::error::Result;
use crate::pipeline::stages::PreIselPrepared;
use crate::pipeline::{FunctionPass, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{OperandConstraintStage, TargetLowering};

pub struct PreIselPass<'a> {
    lowering: &'a dyn TargetLowering,
}

impl<'a> PreIselPass<'a> {
    pub fn new(lowering: &'a dyn TargetLowering) -> Self {
        Self { lowering }
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
        mut mfunc: crate::mir::MachineFunction<PreIselPrepared>,
        ctx: &mut FunctionPassContext<'_, PreIselPrepared>,
    ) -> Result<(crate::mir::MachineFunction<PreIselPrepared>, PassEffect)> {
        for pass in self.lowering.pre_isel_passes() {
            let effect = pass.run(&mut mfunc, ctx)?;
            Self::apply_effect(effect, ctx);
        }

        let pass = crate::passes::constraints::OperandConstraintPass::new(
            self.lowering,
            OperandConstraintStage::PreSelect,
        );
        let effect = FunctionPass::run(&pass, &mut mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        // Effects are applied incrementally above so inner passes see fresh analyses.
        Ok((mfunc, PassEffect::NONE))
    }
}
