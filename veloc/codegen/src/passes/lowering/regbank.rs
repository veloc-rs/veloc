use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::regalloc::RegisterBankSelector;

pub struct RegisterBankSelectionPass;

impl FunctionPass for RegisterBankSelectionPass {
    fn name(&self) -> &'static str {
        "regbank-selected"
    }

    fn run(
        &self,
        mfunc: &mut veloc_lir::MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let changed = RegisterBankSelector::new().select(mfunc, ctx.target);
        let effect = if changed {
            PassEffect::new(ChangeSet::VREG_BANKS)
        } else {
            PassEffect::NONE
        };
        Ok(effect)
    }
}
