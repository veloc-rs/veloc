pub mod abi;
pub mod legalize;
pub(crate) mod reassociate;

use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::TargetLegalizer;
use veloc_lir::MachineFunction;

pub use abi::AbiLoweringPass;
pub use legalize::{LegalizeAction, LegalizeResult, Legalizer};

pub struct LegalizePass<'a> {
    legalizer: &'a dyn TargetLegalizer,
}

impl<'a> LegalizePass<'a> {
    pub fn new(legalizer: &'a dyn TargetLegalizer) -> Self {
        Self { legalizer }
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
        let changed = legalizer.legalize(mfunc)?;
        ctx.stats.legalized_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Ok(if changed {
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::CFG)
        } else {
            PassEffect::NONE
        })
    }
}
