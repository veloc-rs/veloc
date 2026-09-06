//! Function-level optimization passes.

pub mod dce;
pub mod simplify;

pub use dce::DcePass;
pub use simplify::SimplifyPass;
