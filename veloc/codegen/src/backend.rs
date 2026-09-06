//! Backend facade for embedding codegen in upper layers.
//!
//! `veloc-wasm` 仍然倾向于依赖一个稳定的 `Backend` 门面，而不是直接操作
//! `TargetMachine + CodegenPipeline`。这里提供一个轻量适配层，默认使用 x86_64 ELF。

use crate::driver::CodegenPipeline;
use crate::error::{Error, Result};
use crate::target::arch::{TargetConfig, TargetMachine};
use alloc::boxed::Box;
use veloc_mir::Module;

pub struct Backend {
    target: Box<dyn TargetMachine>,
}

impl Backend {
    /// 创建默认 backend。
    ///
    /// 当前默认使用 x86_64 目标，并输出 ELF relocatable object。
    pub fn new() -> Self {
        Self::with_target_config(TargetConfig::default())
            .expect("default codegen target should be available")
    }

    /// 使用显式目标配置创建 backend。
    pub fn with_target_config(config: TargetConfig) -> Result<Self> {
        let arch = config.arch;
        let target = crate::create_target_machine(config)
            .ok_or_else(|| Error::target_machine_unavailable(arch))?;
        Ok(Self { target })
    }

    /// 返回底层目标机。
    pub fn target(&self) -> &dyn TargetMachine {
        &*self.target
    }

    /// 将整个 IR 模块编译为一个 relocatable object。
    pub fn compile_object(&self, module: &Module) -> Result<alloc::vec::Vec<u8>> {
        CodegenPipeline::new(self.target()).compile_object(module)
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::new()
    }
}
