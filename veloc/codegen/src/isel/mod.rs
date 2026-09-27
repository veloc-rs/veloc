pub(crate) mod matching;
pub mod select;

pub use self::select::*;
use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::TargetInstructionSelector;

pub struct InstructionSelectionPass<'a> {
    selector: &'a dyn TargetInstructionSelector,
}

impl<'a> InstructionSelectionPass<'a> {
    pub fn new(selector: &'a dyn TargetInstructionSelector) -> Self {
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
        _ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        select::InstructionSelector::new(self.selector).select(mfunc)?;
        Ok(PassEffect::new(
            ChangeSet::SELECTED_OPCODES | ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
        ))
    }
}
