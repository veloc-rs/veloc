use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::TargetPostIsel;

pub struct PostIselOptimizePass<'a> {
    post_isel: &'a dyn TargetPostIsel,
}

impl<'a> PostIselOptimizePass<'a> {
    pub fn new(post_isel: &'a dyn TargetPostIsel) -> Self {
        Self { post_isel }
    }
}

impl<'a> FunctionPass for PostIselOptimizePass<'a> {
    fn name(&self) -> &'static str {
        "post-isel-optimized"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        if cx.options.optimize {
            self.post_isel.combine_instructions(&mut cx.edit());
        }
        Ok(())
    }
}
