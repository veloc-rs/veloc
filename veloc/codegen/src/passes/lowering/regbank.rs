use crate::error::Result;
use crate::pipeline::stages::{LegalizedLir, PreIselPrepared};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::regalloc::RegisterBankSelector;

pub struct RegisterBankSelectionPass;

impl StageTransformPass<LegalizedLir, PreIselPrepared> for RegisterBankSelectionPass {
    fn name(&self) -> &'static str {
        "regbank-selected"
    }

    fn run(
        &self,
        mut mfunc: crate::lir::MachineFunction<LegalizedLir>,
        ctx: &mut FunctionPassContext<'_, LegalizedLir>,
    ) -> Result<(crate::lir::MachineFunction<PreIselPrepared>, PassEffect)> {
        let changed = RegisterBankSelector::new().select(&mut mfunc, ctx.target);
        let effect = if changed {
            PassEffect::new(ChangeSet::VREG_BANKS)
        } else {
            PassEffect::NONE
        };
        Ok((mfunc.into_stage(), effect))
    }
}
