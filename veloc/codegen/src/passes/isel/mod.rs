pub mod select;

pub use self::select::*;
use crate::error::Result;
use crate::pipeline::stages::{PreIselPrepared, SelectedLir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::TargetInstructionSelector;

pub struct InstructionSelectionPass<'a> {
    selector: &'a dyn TargetInstructionSelector,
}

impl<'a> InstructionSelectionPass<'a> {
    pub fn new(selector: &'a dyn TargetInstructionSelector) -> Self {
        Self { selector }
    }
}

impl<'a> StageTransformPass<PreIselPrepared, SelectedLir> for InstructionSelectionPass<'a> {
    fn name(&self) -> &'static str {
        "selected"
    }

    fn run(
        &self,
        mut mfunc: crate::lir::MachineFunction<PreIselPrepared>,
        ctx: &mut FunctionPassContext<'_, PreIselPrepared>,
    ) -> Result<(crate::lir::MachineFunction<SelectedLir>, PassEffect)> {
        select::InstructionSelector::new(self.selector).select(&mut mfunc)?;
        ctx.stats.selected_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Ok((
            mfunc.into_stage(),
            PassEffect::new(
                ChangeSet::SELECTED_OPCODES | ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
            ),
        ))
    }
}
