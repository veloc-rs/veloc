//! Optimization passes organized by category.

pub mod affine;
pub mod bits;
pub mod cfg;
pub mod cse;
pub mod dce;
pub mod expression;
pub mod inline;
pub mod licm;
pub mod load_cse;
pub mod loop_memory;
pub mod memory;
pub mod memory_validity;
pub mod params;
pub mod partial_inline;
pub mod predicates;
pub mod promote;
pub mod rotate;
pub mod sccp;
pub mod simplify;
pub mod strength;
pub mod threading;

pub use dce::DcePass;
pub use expression::ExpressionPass;
pub use memory::MemoryPass;
pub use simplify::SimplifyPass;
