use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::TargetFrameLowering;

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

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Allocated
    }
    fn output_stage(&self) -> FunctionStage {
        FunctionStage::Framed
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        crate::verify::verify_call_frames(cx.function(), cx.target)?;
        let abi = cx.target.resolve_abi(cx.signature.call_conv)?;
        let mut function = cx.edit();
        self.frame_lowering
            .finalize_stack_frame(&mut function, abi)?;
        self.frame_lowering.insert_prologue_epilogue(&mut function);
        Ok(())
    }
}
