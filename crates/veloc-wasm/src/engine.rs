use std::path::PathBuf;
use std::sync::Arc;

use veloc::codegen::Backend;

/// 编译策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Strategy {
    #[default]
    Auto,
    Jit,
    Interpreter,
}

/// Engine 配置
#[derive(Debug, Clone)]
pub struct Config {
    pub codegen: veloc::codegen::CodegenOptions,
    pub strategy: Strategy,
    pub dump_ir: bool,
    pub ir_names: bool,
    /// Validate translated MIR before optimization and code generation.
    pub verify_ir: bool,
    /// 优化等级
    pub opt_level: u8,
    /// Use a smaller equality-search budget in MIR expression optimization.
    pub fast_egraph: bool,
    /// 输出 IR 到文件路径
    pub output_ir: Option<PathBuf>,
    /// Chrome Trace 输出文件路径
    pub trace_file: Option<PathBuf>,
    /// 是否打印优化统计信息
    pub print_stats: bool,
    /// 优化调试标签
    pub opt_debug: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            codegen: Default::default(),
            strategy: Strategy::Auto,
            dump_ir: false,
            ir_names: false,
            verify_ir: cfg!(debug_assertions),
            opt_level: 0,
            fast_egraph: false,
            output_ir: None,
            trace_file: None,
            print_stats: false,
            opt_debug: Vec::new(),
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
