use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
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

impl<'a> FunctionPass for PostIselOptimizePass<'a> {
    fn name(&self) -> &'static str {
        "post-isel-optimized"
    }

    fn run(
        &self,
        mfunc: &mut veloc_lir::MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        self.post_isel.combine_instructions(mfunc);
        ctx.stats.combined_inst_count = mfunc.blocks().map(|b| mfunc.block_insts(b).count()).sum();
        crate::passes::constraints::PostSelectOperandConstraintPass::new(self.operand_lowering)
            .run(mfunc)?;
        Ok(PassEffect::new(
            ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
        ))
    }
}
