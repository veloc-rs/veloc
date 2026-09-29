//! Optimization passes organized by category.

pub mod dce;
pub mod expression;
pub mod memory;
pub mod simplify;

pub use dce::DcePass;
pub use expression::ExpressionPass;
pub use memory::MemoryPass;
pub use simplify::SimplifyPass;
