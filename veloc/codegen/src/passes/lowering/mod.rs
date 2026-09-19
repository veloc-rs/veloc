pub mod abi;
pub(crate) mod control;
pub mod legalize;

use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::TargetLegalizer;
use veloc_lir::MachineFunction;

pub use abi::AbiLoweringPass;
pub use legalize::{LegalizeAction, Legalizer, RewriteContext};

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
        let changed = legalizer.legalize(mfunc, |f, id| abi::lower_call(ctx.target, f, id))?;
        Ok(if changed {
            PassEffect::new(
                ChangeSet::INST_SEMANTICS
                    | ChangeSet::CFG
                    | ChangeSet::STACK_FRAME
                    | ChangeSet::PHYSICAL_REGS,
            )
        } else {
            PassEffect::NONE
        })
    }
}
