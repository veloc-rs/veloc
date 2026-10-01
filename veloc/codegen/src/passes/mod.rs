pub mod frame;
pub mod isel;
pub mod lowering;
pub mod postisel;
pub mod unreachable;

pub use frame::FrameFinalizePass;
pub use isel::InstructionSelectionPass;
pub use lowering::LegalizePass;
pub use postisel::PostIselOptimizePass;
pub use unreachable::RemoveUnreachablePass;
pub mod schedule;
pub(crate) mod state;
