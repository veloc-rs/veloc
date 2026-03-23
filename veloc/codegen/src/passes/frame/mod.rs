use crate::error::Result;
use crate::pipeline::stages::{PrologueEpilogueInserted, RegAllocated};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::CallConv;

pub struct FrameFinalizePass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> FrameFinalizePass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<RegAllocated, PrologueEpilogueInserted> for FrameFinalizePass<'a> {
    fn name(&self) -> &'static str {
        "frame-finalized"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<RegAllocated>,
        ctx: &mut FunctionPassContext<'_, RegAllocated>,
    ) -> Result<(
        crate::mir::MachineFunction<PrologueEpilogueInserted>,
        PassEffect,
    )> {
        self.lowering.finalize_stack_frame(
            mfunc.as_untyped_mut(),
            CallConv::from(ctx.func_sig.call_conv),
        );
        self.lowering
            .insert_prologue_epilogue(mfunc.as_untyped_mut());
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::BLOCK_LAYOUT | ChangeSet::STACK_FRAME),
        ))
    }
}
