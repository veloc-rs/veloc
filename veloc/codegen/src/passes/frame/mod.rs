use crate::error::Result;
use crate::pipeline::stages::{PrologueEpilogueInserted, RegAllocated};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{CallConv, TargetFrameLowering};

pub struct FrameFinalizePass<'a> {
    frame_lowering: &'a dyn TargetFrameLowering,
}

impl<'a> FrameFinalizePass<'a> {
    pub fn new(frame_lowering: &'a dyn TargetFrameLowering) -> Self {
        Self { frame_lowering }
    }
}

impl<'a> StageTransformPass<RegAllocated, PrologueEpilogueInserted> for FrameFinalizePass<'a> {
    fn name(&self) -> &'static str {
        "frame-finalized"
    }

    fn run(
        &self,
        mut mfunc: crate::lir::MachineFunction<RegAllocated>,
        ctx: &mut FunctionPassContext<'_, RegAllocated>,
    ) -> Result<(
        crate::lir::MachineFunction<PrologueEpilogueInserted>,
        PassEffect,
    )> {
        self.frame_lowering
            .finalize_stack_frame(&mut mfunc, CallConv::from(ctx.func_sig.call_conv));
        self.frame_lowering.insert_prologue_epilogue(&mut mfunc);
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::BLOCK_LAYOUT | ChangeSet::STACK_FRAME),
        ))
    }
}
