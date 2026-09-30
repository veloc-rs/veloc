pub mod matching;
pub mod select;

pub use self::select::*;
use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};

pub struct InstructionSelectionPass<'a> {
    selector: SelectPolicy<'a>,
}

impl<'a> InstructionSelectionPass<'a> {
    pub fn new(selector: SelectPolicy<'a>) -> Self {
        Self { selector }
    }
}

impl<'a> FunctionPass for InstructionSelectionPass<'a> {
    fn name(&self) -> &'static str {
        "selected"
    }

    fn run(
        &self,
        mfunc: &mut veloc_lir::MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let cfg = ctx.function_analyses.cfg(mfunc, ctx.target);
        select::InstructionSelector::new(self.selector).select(mfunc, cfg)?;
        Ok(PassEffect::new(
            ChangeSet::SELECTED_OPCODES | ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
        ))
    }
}
