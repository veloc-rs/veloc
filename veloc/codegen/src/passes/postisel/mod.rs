use crate::error::Result;
use crate::pipeline::stages::{PostIselOptimized, SelectedMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{TargetOperandLowering, TargetPostIsel};

pub struct PostIselOptimizePass<'a> {
    post_isel: &'a dyn TargetPostIsel,
    operand_lowering: &'a dyn TargetOperandLowering,
}

impl<'a> PostIselOptimizePass<'a> {
    pub fn new(
        post_isel: &'a dyn TargetPostIsel,
        operand_lowering: &'a dyn TargetOperandLowering,
    ) -> Self {
        Self {
            post_isel,
            operand_lowering,
        }
    }
}

impl<'a> StageTransformPass<SelectedMir, PostIselOptimized> for PostIselOptimizePass<'a> {
    fn name(&self) -> &'static str {
        "post-isel-optimized"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<SelectedMir>,
        ctx: &mut FunctionPassContext<'_, SelectedMir>,
    ) -> Result<(crate::mir::MachineFunction<PostIselOptimized>, PassEffect)> {
        self.post_isel.combine_instructions(&mut mfunc);
        ctx.stats.combined_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        crate::passes::constraints::PostSelectOperandConstraintPass::new(self.operand_lowering)
            .run(&mut mfunc)?;
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS),
        ))
    }
}
