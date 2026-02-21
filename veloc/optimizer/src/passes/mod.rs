//! Optimization passes organized by category.

pub mod function;
pub mod r#loop;
pub mod module;

pub use function::{ConstantFoldingPass, DcePass};
