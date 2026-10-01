pub mod abi;
pub(crate) mod control;
pub mod legalize;

use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use legalize::LegalizePolicy;

pub use abi::AbiLoweringPass;
pub use legalize::{Legalizer, RewriteContext};

pub struct LegalizePass<'a> {
    legalizer: LegalizePolicy<'a>,
}

impl<'a> LegalizePass<'a> {
    pub fn new(legalizer: LegalizePolicy<'a>) -> Self {
        Self { legalizer }
    }
}

impl<'a> FunctionPass for LegalizePass<'a> {
    fn name(&self) -> &'static str {
        "legalize"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Generic
    }
    fn output_stage(&self) -> FunctionStage {
        FunctionStage::Legal
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let target = cx.target;
        let mut edit = cx.edit();
        let (function, symbols) = edit.with_symbols();
        Legalizer::new(self.legalizer).legalize(function, target, symbols)?;
        Ok(())
    }
}
