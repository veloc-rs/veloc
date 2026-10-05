extern crate alloc;

mod error;
pub mod evaluate;
pub mod manager;
pub mod pass;
pub mod passes;
mod rewrite;

pub use error::Error;
pub use manager::PassManager;
pub use pass::{FunctionPass, ModulePass, OptConfig, PassOutcome};
pub use passes::{DcePass, ExpressionPass, SimplifyPass};
pub use veloc_profile::Profile;

/// 获取所有已知的调试标签列表
fn get_known_debug_tags() -> &'static [&'static str] {
    &["dce", "simplify", "liveness", "gvn", "inline"]
}

type Result<T> = core::result::Result<T, Error>;
