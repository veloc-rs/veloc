pub mod abi;
pub mod block_params;
pub(crate) mod generic_egraph;
pub mod legalize;
pub mod regbank;

use crate::error::Result;
use crate::mir::MachineFunction;
use crate::pipeline::stages::{LegalizedMir, PreIselPrepared};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::TargetLowering;

use self::generic_egraph::run_generic_pre_isel_egraph_combine;

pub use abi::AbiLoweringPass;
pub use block_params::BlockParamLoweringPass;
pub use legalize::{Legalizer, LegalizerInfo};
pub use regbank::RegisterBankSelectionPass;

pub struct LegalizePass<'a> {
    lowering: &'a dyn TargetLowering,
}

impl<'a> LegalizePass<'a> {
    pub fn new(lowering: &'a dyn TargetLowering) -> Self {
        Self { lowering }
    }

    fn apply_effect(effect: PassEffect, ctx: &mut FunctionPassContext<'_, LegalizedMir>) {
        if !effect.change_set.is_empty() {
            ctx.function_analyses.apply(effect.change_set);
        }
    }
}

impl<'a> StageTransformPass<LegalizedMir, PreIselPrepared> for LegalizePass<'a> {
    fn name(&self) -> &'static str {
        "legalize"
    }

    fn run(
        &self,
        mut mfunc: MachineFunction<LegalizedMir>,
        ctx: &mut FunctionPassContext<'_, LegalizedMir>,
    ) -> Result<(MachineFunction<PreIselPrepared>, PassEffect)> {
        let legalizer = Legalizer::new(self.lowering.legalizer_info(), self.lowering);
        legalizer.legalize(mfunc.as_untyped_mut());
        ctx.stats.legalized_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Self::apply_effect(
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::CFG),
            ctx,
        );

        for pass in self.lowering.post_legalize_passes() {
            let effect = pass.run(&mut mfunc, ctx)?;
            Self::apply_effect(effect, ctx);
        }

        run_generic_pre_isel_egraph_combine(mfunc.as_untyped_mut(), ctx.function_analyses);

        let abi = AbiLoweringPass::new();
        let (mfunc, effect) = abi.run(mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        let regbank = RegisterBankSelectionPass;
        let (mfunc, effect) = regbank.run(mfunc, ctx)?;
        Self::apply_effect(effect, ctx);

        // Effects are applied incrementally above so nested passes observe fresh analyses.
        Ok((mfunc, PassEffect::NONE))
    }
}
