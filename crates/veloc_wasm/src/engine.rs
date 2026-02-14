use std::sync::Arc;

use veloc::backend::Backend;

/// 编译策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Auto,
    Jit,
    Interpreter,
}

/// Engine 配置
#[derive(Debug, Clone)]
pub struct Config {
    pub strategy: Strategy,
    pub dump_ir: bool,
    pub ir_names: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            strategy: Strategy::Auto,
            dump_ir: false,
            ir_names: false,
        }
    }
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    backend: Backend,
    config: Config,
}

impl Engine {
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Self {
        Self {
            inner: Arc::new(EngineInner {
                backend: Backend::new(),
                config,
            }),
        }
    }

    pub fn strategy(&self) -> Strategy {
        self.inner.config.strategy
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    pub(crate) fn backend(&self) -> &Backend {
        &self.inner.backend
    }
}
