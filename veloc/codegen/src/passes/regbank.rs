use crate::error::Result;
use crate::pipeline::stages::{AbiLowered, BankSelected};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::regalloc::RegisterBankSelector;

pub struct RegisterBankSelectionPass;

impl StageTransformPass<AbiLowered, BankSelected> for RegisterBankSelectionPass {
    fn name(&self) -> &'static str {
        "regbank-selected"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<AbiLowered>,
        ctx: &mut FunctionPassContext<'_, AbiLowered>,
    ) -> Result<(crate::mir::MachineFunction<BankSelected>, PassEffect)> {
        RegisterBankSelector::new().select(mfunc.as_untyped_mut(), ctx.target);
        Ok((mfunc.into_stage(), PassEffect::new(ChangeSet::VREG_BANKS)))
    }
}
