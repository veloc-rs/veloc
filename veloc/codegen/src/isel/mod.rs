pub mod matching;
pub mod select;

pub use self::select::*;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};

pub struct InstructionSelectionPass<'a> {
    selector: SelectPolicy<'a>,
}

impl<'a> InstructionSelectionPass<'a> {
    pub fn new(selector: SelectPolicy<'a>) -> Self {
        Self { selector }
    }
}

impl<'a> FunctionPass for InstructionSelectionPass<'a> {
    fn name(&self) -> &'static str {
        "selected"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Legal
    }
    fn output_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let entry = cx.function().entry_block();
        let blocks = cx.cfg().compute_post_order(entry);
        select::InstructionSelector::new(self.selector).select_in_order(&mut cx.edit(), blocks)
    }
}
