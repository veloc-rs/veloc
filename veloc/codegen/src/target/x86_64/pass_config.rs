use crate::target::{TargetPassConfig, TargetPostIsel};
use std::vec::Vec;

#[derive(Debug, Clone, Copy)]
pub struct X86_64PostIsel;

#[derive(Debug, Clone, Copy)]
pub struct X86_64PassConfig;

impl TargetPostIsel for X86_64PostIsel {}

impl TargetPassConfig for X86_64PassConfig {
    fn prepare_passes(&self) -> Vec<std::boxed::Box<dyn crate::pipeline::FunctionPass>> {
        // Policy: use a comparison chain until jump-table selection is available.
        std::vec![std::boxed::Box::new(
            crate::passes::lowering::control::BranchTableLowering
        )]
    }
}
