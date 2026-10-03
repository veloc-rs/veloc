use crate::target::TargetPassConfig;
use std::vec::Vec;

#[derive(Debug, Clone, Copy)]
pub struct X86_64PassConfig;

impl TargetPassConfig for X86_64PassConfig {
    fn prepare_passes(
        &self,
        _level: crate::OptLevel,
    ) -> Vec<std::boxed::Box<dyn crate::pipeline::FunctionPass>> {
        // Policy: use a comparison tree until jump-table selection is available.
        std::vec![std::boxed::Box::new(
            crate::passes::lowering::control::BranchTableLowering {
                max_cases: usize::MAX
            }
        )]
    }
}
