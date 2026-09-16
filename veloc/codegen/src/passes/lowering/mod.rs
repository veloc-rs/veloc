pub mod abi;
pub mod legalize;
pub(crate) mod reassociate;
pub mod regbank;

use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{TargetLegalizer, TargetPassConfig};
use veloc_lir::MachineFunction;

use self::reassociate::reassociate;

pub use abi::AbiLoweringPass;
pub use legalize::{LegalizeAction, LegalizeResult, Legalizer};
pub use regbank::RegisterBankSelectionPass;

pub struct LegalizePass<'a> {
    legalizer: &'a dyn TargetLegalizer,
    pass_config: &'a dyn TargetPassConfig,
}

impl<'a> LegalizePass<'a> {
    pub fn new(legalizer: &'a dyn TargetLegalizer, pass_config: &'a dyn TargetPassConfig) -> Self {
        Self {
            legalizer,
            pass_config,
        }
    }

    fn apply_effect(effect: PassEffect, ctx: &mut FunctionPassContext<'_>) {
        if !effect.change_set.is_empty() {
            ctx.function_analyses.apply(effect.change_set);
        }
    }
}

impl<'a> FunctionPass for LegalizePass<'a> {
    fn name(&self) -> &'static str {
        "legalize"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let legalizer = Legalizer::new(self.legalizer);
        legalizer.legalize(mfunc)?;
        ctx.stats.legalized_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Self::apply_effect(
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::CFG),
            ctx,
        );

        for pass in self.pass_config.post_legalize_passes() {
            let effect = pass.run(mfunc, ctx)?;
            Self::apply_effect(effect, ctx);
        }

        reassociate(mfunc, ctx.function_analyses);

        let abi = AbiLoweringPass::new();
        let effect = abi.run(mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        let regbank = RegisterBankSelectionPass;
        let effect = regbank.run(mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        // Effects are applied incrementally above so nested passes observe fresh analyses.
        Ok(PassEffect::NONE)
    }
}
