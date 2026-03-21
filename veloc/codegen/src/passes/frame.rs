use crate::error::Result;
use crate::pipeline::stages::{FrameFinalized, PrologueEpilogueInserted, RegAllocated};
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

impl<'a> StageTransformPass<RegAllocated, FrameFinalized> for FrameFinalizePass<'a> {
    fn name(&self) -> &'static str {
        "frame-finalized"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<RegAllocated>,
        ctx: &mut FunctionPassContext<'_, RegAllocated>,
    ) -> Result<(crate::mir::MachineFunction<FrameFinalized>, PassEffect)> {
        self.lowering.finalize_stack_frame(
            mfunc.as_untyped_mut(),
            CallConv::from(ctx.func_sig.call_conv),
        );
        Ok((mfunc.into_stage(), PassEffect::new(ChangeSet::STACK_FRAME)))
    }
}

pub struct PrologueEpiloguePass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> PrologueEpiloguePass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<FrameFinalized, PrologueEpilogueInserted> for PrologueEpiloguePass<'a> {
    fn name(&self) -> &'static str {
        "prologue-epilogue"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<FrameFinalized>,
        _ctx: &mut FunctionPassContext<'_, FrameFinalized>,
    ) -> Result<(
        crate::mir::MachineFunction<PrologueEpilogueInserted>,
        PassEffect,
    )> {
        self.lowering
            .insert_prologue_epilogue(mfunc.as_untyped_mut());
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::BLOCK_LAYOUT | ChangeSet::STACK_FRAME),
        ))
    }
}
