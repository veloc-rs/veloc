//! Target-specific Code Generation
//!
//! 提供多架构支持的代码生成框架

pub mod arch;
pub mod x86_64;

pub use arch::*;
