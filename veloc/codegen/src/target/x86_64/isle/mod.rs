//! Generated ISLE for x86_64
//!
//! 此模块包含由 ISLE 编译器自动生成的指令选择和发射代码。

pub mod generated {
    // build.rs 将生成的文件放在 OUT_DIR 中
    include!(concat!(env!("OUT_DIR"), "/isle_x86_64.rs"));
}

pub use generated::*;
