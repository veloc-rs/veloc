use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
pub struct PostIselOptimizePass;

impl FunctionPass for PostIselOptimizePass {
    fn name(&self) -> &'static str {
        "post-isel-optimized"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let target = cx.target;
        target.post_isel().combine_instructions(&mut cx.edit());
        Ok(())
    }
}
