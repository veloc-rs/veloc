//! Function-level optimization passes.

pub mod dce;
pub mod expression;
pub mod memory;

pub use dce::DcePass;
pub use expression::ExpressionPass;
pub use memory::MemoryPass;
