//! Function-level optimization passes.

pub mod dce;
pub mod memory;
pub mod simplify;

pub use dce::DcePass;
pub use memory::MemoryPass;
pub use simplify::SimplifyPass;
