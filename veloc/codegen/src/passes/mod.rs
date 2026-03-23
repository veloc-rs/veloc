pub mod constraints;
pub mod frame;
pub mod isel;
pub mod lowering;
pub mod postisel;
pub mod preisel;
pub mod regalloc;

pub use frame::FrameFinalizePass;
pub use isel::InstructionSelectionPass;
pub use lowering::{BlockParamLoweringPass, LegalizePass};
pub use postisel::PostIselOptimizePass;
pub use preisel::PreIselPass;
pub use regalloc::RegisterAllocationPass;
