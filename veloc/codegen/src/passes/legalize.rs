use crate::error::Result;
use crate::isel::Legalizer;
use crate::mir::MachineFunction;
use crate::pipeline::stages::{BlockParamsLowered, LegalizedMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::TargetLowering;

pub struct LegalizePass<'a> {
    lowering: &'a dyn TargetLowering,
}

impl<'a> LegalizePass<'a> {
    pub fn new(lowering: &'a dyn TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<BlockParamsLowered, LegalizedMir> for LegalizePass<'a> {
    fn name(&self) -> &'static str {
        "legalized"
    }

    fn run(
        &self,
        mut mfunc: MachineFunction<BlockParamsLowered>,
        ctx: &mut FunctionPassContext<'_, BlockParamsLowered>,
    ) -> Result<(MachineFunction<LegalizedMir>, PassEffect)> {
        let legalizer = Legalizer::new(self.lowering.legalizer_info(), self.lowering);
        legalizer.legalize(mfunc.as_untyped_mut());
        ctx.stats.legalized_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::CFG),
        ))
    }
}
