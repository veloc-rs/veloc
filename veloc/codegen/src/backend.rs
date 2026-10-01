//! Owns the configured target machine for upper layers.

use crate::error::Result;
use crate::target::{TargetConfig, TargetMachine};
use std::boxed::Box;

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
        let target = crate::create_target_machine(config)?;
        Ok(Self { target })
    }

    /// 返回底层目标机。
    pub fn target(&self) -> &dyn TargetMachine {
        &*self.target
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::new()
    }
}
