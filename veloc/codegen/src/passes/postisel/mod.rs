use crate::error::Result;
use crate::pipeline::stages::{PostIselOptimized, SelectedMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};

pub struct PostIselOptimizePass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> PostIselOptimizePass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
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
        self.lowering.combine_instructions(&mut mfunc);
        ctx.stats.combined_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        crate::passes::constraints::PostSelectOperandConstraintPass::new(self.lowering)
            .run(&mut mfunc)?;
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS),
        ))
    }
}
