pub mod select;

pub use self::select::*;
use crate::error::Result;
use crate::pipeline::stages::{PreIselPrepared, SelectedMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};

pub struct InstructionSelectionPass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> InstructionSelectionPass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<PreIselPrepared, SelectedMir> for InstructionSelectionPass<'a> {
    fn name(&self) -> &'static str {
        "selected"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<PreIselPrepared>,
        ctx: &mut FunctionPassContext<'_, PreIselPrepared>,
    ) -> Result<(crate::mir::MachineFunction<SelectedMir>, PassEffect)> {
        select::InstructionSelector::new(self.lowering).select(&mut mfunc)?;
        ctx.stats.selected_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Ok((
            mfunc.into_stage(),
            PassEffect::new(
                ChangeSet::SELECTED_OPCODES | ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
            ),
        ))
    }
}
