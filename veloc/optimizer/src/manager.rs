use crate::pass::{FunctionPass, ModulePass, OptConfig, Pass, PreservedAnalyses};
use crate::passes::function::dce;
use crate::stats::{PipelineStats, TimingGuard};
use alloc::boxed::Box;
use alloc::vec::Vec;
use std::time::Instant;
use veloc_analyzer::AnalysisManager;
use veloc_ir::{ModuleData, function::Function};

/// 优化流程管理器。
pub struct PassManager {
    passes: Vec<Pass>,
    am: AnalysisManager,
    config: OptConfig,
    pub stats: PipelineStats,
}

impl PassManager {
    pub fn new(config: OptConfig) -> Self {
        Self {
            passes: Vec::new(),
            am: AnalysisManager::new(),
            config,
            stats: PipelineStats::default(),
        }
    }

    pub fn new_o1() -> Self {
        let mut pm = Self::new(OptConfig::new(true));
        pm.add_function_pass(dce::DcePass);
        pm
    }

    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    pub fn config(&self) -> &OptConfig {
        &self.config
    }

    pub fn add_function_pass<P: FunctionPass + 'static>(&mut self, pass: P) {
        self.passes.push(Pass::Function(Box::new(pass)));
    }

    pub fn add_module_pass<P: ModulePass + 'static>(&mut self, pass: P) {
        self.passes.push(Pass::Module(Box::new(pass)));
    }

    /// 在整个模块上运行所有 Pass。
    pub fn run_on_module(&mut self, module: &mut ModuleData) -> bool {
        let mut changed = false;
        self.stats.start_session();
        let total_start = Instant::now();

        for pass in &self.passes {
            let guard = if self.config.monitor_performance {
                Some(TimingGuard::new(pass.name()))
            } else {
                None
            };

            let pa = match pass {
                Pass::Module(mp) => {
                    let pa = mp.run(module, &mut self.am, &self.config, &mut self.stats.metrics);
                    if pa.changed() {
                        changed = true;
                        module.bump_revision();
                    }
                    pa
                }
                Pass::Function(fp) => {
                    let mut fp_changed = false;
                    for (_, func) in module.functions.iter_mut() {
                        let pa = fp.run(func, &mut self.am, &self.config, &mut self.stats.metrics);
                        if pa.changed() {
                            fp_changed = true;
                            func.bump_revision();
                        }
                    }
                    if fp_changed {
                        changed = true;
                    }
                    PreservedAnalyses::none()
                }
            };

            if let Some(g) = guard {
                g.finish(&mut self.stats);
            }

            self.am
                .invalidate_with_preserved(|id| pa.is_preserved_id(id));
        }

        self.stats.total_duration = total_start.elapsed();
        if self.config.monitor_performance {
            println!("{}", self.stats);
        }
        changed
    }

    /// 单独在某个函数上运行已注册的操作。
    pub fn run_on_function(&mut self, func: &mut Function) -> bool {
        let mut changed = false;
        self.stats.start_session();
        let total_start = Instant::now();

        for pass in &self.passes {
            match pass {
                Pass::Function(fp) => {
                    let guard = if self.config.monitor_performance {
                        Some(TimingGuard::new(fp.name()))
                    } else {
                        None
                    };

                    let pa = fp.run(func, &mut self.am, &self.config, &mut self.stats.metrics);
                    if pa.changed() {
                        changed = true;
                        func.bump_revision();
                    }

                    if let Some(g) = guard {
                        g.finish(&mut self.stats);
                    }

                    self.am
                        .invalidate_with_preserved(|id| pa.is_preserved_id(id));
                }
                Pass::Module(_) => {}
            }
        }

        self.stats.total_duration = total_start.elapsed();
        if self.config.monitor_performance {
            println!("{}", self.stats);
        }
        changed
    }
}
