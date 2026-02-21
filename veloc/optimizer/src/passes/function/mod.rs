//! Function-level optimization passes.

pub mod constant_folding;
pub mod dce;

pub use constant_folding::ConstantFoldingPass;
pub use dce::DcePass;
