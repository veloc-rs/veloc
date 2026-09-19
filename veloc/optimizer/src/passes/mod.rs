//! Optimization passes organized by category.

pub mod function;

pub use function::{DcePass, ExpressionPass};
