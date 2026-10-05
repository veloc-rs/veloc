use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{Error, Result};
use veloc::codegen::Backend;
pub use veloc::codegen::OptLevel;

/// 编译策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Strategy {
    #[default]
    Auto,
    Jit,
    FastJit,
    Interpreter,
}

/// Bounds-check implementation. Guarded objects require a compatible runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum MemoryChecks {
    /// Use guard pages for supported native execution, software checks otherwise.
    #[default]
    Auto,
    Software,
    Guarded,
}

/// Engine 配置
#[derive(Debug, Clone)]
pub struct Config {
    /// The optimization level selects both MIR and native-code pipelines.
    pub codegen: veloc::codegen::CodegenOptions,
    /// Backend architecture for native compilation. Execution requires the host architecture.
    pub target: veloc::codegen::TargetArch,
    /// CPU model and ISA overrides for the selected native backend.
    pub cpu: String,
    pub cpu_features: Vec<String>,
    pub strategy: Strategy,
    /// Native guard pages are supported on Linux x86-64 and RV64 with glibc;
    /// interpreter guard pages are supported on Linux x86-64 with glibc.
    pub memory_checks: MemoryChecks,
    pub dump_ir: bool,
    pub ir_names: bool,
    /// Validate translated MIR before optimization and code generation.
    pub verify_ir: bool,
    /// Use a smaller equality-search budget in MIR expression optimization.
    pub fast_egraph: bool,
    /// Chrome Trace 输出文件路径
    pub trace_file: Option<PathBuf>,
    /// Include bounded optimization remarks and LIR snapshots in the trace.
    pub trace_details: bool,
    /// Print compilation timings and metrics
    pub print_stats: bool,
    /// 优化调试标签
    pub opt_debug: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            codegen: veloc::codegen::CodegenOptions {
                opt_level: OptLevel::None,
                ..Default::default()
            },
            target: if cfg!(target_arch = "riscv64") {
                veloc::codegen::TargetArch::Riscv64
            } else {
                veloc::codegen::TargetArch::X86_64
            },
            cpu: "generic".into(),
            cpu_features: Vec::new(),
            strategy: Strategy::Auto,
            memory_checks: MemoryChecks::Auto,
            dump_ir: false,
            ir_names: false,
            verify_ir: cfg!(debug_assertions),
            fast_egraph: false,
            trace_file: None,
            trace_details: false,
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
    backend: Option<Backend>,
    guarded_memory: bool,
    config: Config,
}

impl Engine {
    pub fn new() -> Result<Self> {
        Self::with_config(Config::default())
    }

    pub fn with_config(mut config: Config) -> Result<Self> {
        if config.strategy == Strategy::Auto {
            config.strategy = Strategy::Jit;
        }
        if config.strategy == Strategy::FastJit
            && (config.target != veloc::codegen::TargetArch::X86_64
                || config.cpu != "generic"
                || !config.cpu_features.is_empty())
        {
            return Err(Error::Unsupported(
                "fast-jit requires generic x86_64 without feature overrides".into(),
            ));
        }
        if config.trace_details && config.trace_file.is_none() {
            return Err(Error::Message("trace details require a trace file".into()));
        }
        if config.fast_egraph && config.codegen.opt_level == OptLevel::None {
            return Err(Error::Message(
                "fast e-graph requires optimization level 1".into(),
            ));
        }
        if !config.opt_debug.is_empty() && config.codegen.opt_level == OptLevel::None {
            return Err(Error::Message(
                "optimizer debug tags require optimization level 1".into(),
            ));
        }
        let guarded_memory = match config.memory_checks {
            MemoryChecks::Auto => {
                config.strategy != Strategy::Interpreter && crate::trap::native::SUPPORTED
            }
            MemoryChecks::Software => false,
            MemoryChecks::Guarded => true,
        };
        let backend = if config.strategy == Strategy::Interpreter {
            None
        } else {
            Some(
                Backend::with_target_config(veloc::codegen::TargetConfig {
                    cpu: config.cpu.clone(),
                    features: config.cpu_features.clone(),
                    arch: config.target,
                    ..Default::default()
                })
                .map_err(|error| Error::Compile(error.to_string()))?,
            )
        };
        Ok(Self {
            inner: Arc::new(EngineInner {
                backend,
                guarded_memory,
                config,
            }),
        })
    }

    pub fn strategy(&self) -> Strategy {
        self.inner.config.strategy
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    pub(crate) fn uses_guarded_memory(&self) -> bool {
        self.inner.guarded_memory
    }

    /// Execution requirements are checked before translation. Emitting an
    /// object is allowed to use a different target from the current host.
    pub(crate) fn validate_execution(&self) -> Result<()> {
        let interpreter = self.strategy() == Strategy::Interpreter;
        if !interpreter {
            let host = if cfg!(target_arch = "x86_64") {
                Some(veloc::codegen::TargetArch::X86_64)
            } else if cfg!(target_arch = "riscv64") {
                Some(veloc::codegen::TargetArch::Riscv64)
            } else {
                None
            };
            if host != Some(self.config().target) {
                return Err(Error::Unsupported("native execution requires the host target; use Module::emit for cross compilation".into()));
            }
            if self.strategy() == Strategy::FastJit
                && !cfg!(all(target_arch = "x86_64", target_os = "linux"))
            {
                return Err(Error::Unsupported(
                    "fast-jit requires an x86_64 Linux host".into(),
                ));
            }
        }
        if self.uses_guarded_memory() {
            let supported = if interpreter {
                cfg!(all(
                    target_arch = "x86_64",
                    target_os = "linux",
                    target_env = "gnu"
                ))
            } else {
                crate::trap::native::SUPPORTED
            };
            if !supported {
                return Err(Error::Unsupported(
                    "guarded memory is unavailable for this execution backend and host".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn data_layout(&self) -> veloc_types::DataLayout {
        self.inner
            .backend
            .as_ref()
            .map_or(veloc::interpreter::DATA_LAYOUT, |b| {
                b.target().desc().data_layout
            })
    }

    pub(crate) fn backend(&self) -> &Backend {
        self.inner
            .backend
            .as_ref()
            .expect("native compilation backend")
    }
}
