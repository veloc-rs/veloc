pub(crate) mod matching;
pub mod select;

pub use self::select::*;
use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::TargetInstructionSelector;

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
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        select::InstructionSelector::new(self.selector).select(mfunc)?;
        ctx.stats.selected_inst_count = mfunc.blocks().map(|b| mfunc.block_insts(b).count()).sum();
        Ok(PassEffect::new(
            ChangeSet::SELECTED_OPCODES | ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
        ))
    }
}
