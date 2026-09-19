use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};
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

    fn run(
        &self,
        mfunc: &mut veloc_lir::MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        if !ctx.options.optimize {
            return Ok(PassEffect::NONE);
        }
        self.post_isel.combine_instructions(mfunc);
        Ok(PassEffect::new(
            ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
        ))
    }
}
