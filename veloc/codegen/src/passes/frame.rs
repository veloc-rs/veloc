use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::{CallConv, TargetFrameLowering};

pub struct FrameFinalizePass<'a> {
    frame_lowering: &'a dyn TargetFrameLowering,
}

impl<'a> FrameFinalizePass<'a> {
    pub fn new(frame_lowering: &'a dyn TargetFrameLowering) -> Self {
        Self { frame_lowering }
    }
}

impl<'a> FunctionPass for FrameFinalizePass<'a> {
    fn name(&self) -> &'static str {
        "frame-finalized"
    }

    fn run(
        &self,
        mfunc: &mut veloc_lir::MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        self.frame_lowering
            .finalize_stack_frame(mfunc, CallConv::from(ctx.func_sig.call_conv));
        self.frame_lowering.insert_prologue_epilogue(mfunc);
        Ok(PassEffect::new(
            ChangeSet::INST_LAYOUT | ChangeSet::STACK_FRAME,
        ))
    }
}
