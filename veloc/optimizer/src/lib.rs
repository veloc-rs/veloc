extern crate alloc;

pub mod dce;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::TypeId;
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_ir::{ModuleData, function::Function};

/// 声明在 Pass 执行后保留的分析结果。
pub struct PreservedAnalyses {
    preserved_ids: HashSet<TypeId>,
    preserve_all: bool,
}

impl PreservedAnalyses {
    pub fn none() -> Self {
        Self {
            preserved_ids: HashSet::new(),
            preserve_all: false,
        }
    }

    pub fn all() -> Self {
        Self {
            preserved_ids: HashSet::new(),
            preserve_all: true,
        }
    }

    pub fn preserve<A: 'static>(&mut self) {
        self.preserved_ids.insert(TypeId::of::<A>());
    }

    pub fn is_preserved_id(&self, id: TypeId) -> bool {
        self.preserve_all || self.preserved_ids.contains(&id)
    }

    pub fn changed(&self) -> bool {
        !self.preserve_all
    }
}

/// 作用于单个函数的优化 Pass。
pub trait FunctionPass {
    fn name(&self) -> &str;
    fn run(
        &self,
        func: &mut Function,
        am: &mut AnalysisManager,
        config: &OptimizationConfig,
    ) -> PreservedAnalyses;
}

/// 作用于整个模块的优化 Pass。
pub trait ModulePass {
    fn name(&self) -> &str;
    fn run(
        &self,
        module: &mut ModuleData,
        am: &mut AnalysisManager,
        config: &OptimizationConfig,
    ) -> PreservedAnalyses;
}

/// 优化 Pass 的类型包装。
pub enum Pass {
    Function(Box<dyn FunctionPass>),
    Module(Box<dyn ModulePass>),
}

/// 优化配置。
#[derive(Debug, Clone, Default)]
pub struct OptimizationConfig {
    pub print_removed_insts: bool,
}

/// 优化流程管理器。
pub struct PassManager {
    passes: Vec<Pass>,
    am: AnalysisManager,
    config: OptimizationConfig,
}

impl PassManager {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            passes: Vec::new(),
            am: AnalysisManager::new(),
            config,
        }
    }

    pub fn new_o1() -> Self {
        let mut pm = Self::new(OptimizationConfig::default());
        pm.add_function_pass(dce::DcePass);
        pm
    }

    pub fn config(&self) -> &OptimizationConfig {
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
        for pass in &self.passes {
            match pass {
                Pass::Module(mp) => {
                    let pa = mp.run(module, &mut self.am, &self.config);
                    if pa.changed() {
                        changed = true;
                        module.bump_revision();
                    }
                    self.am
                        .invalidate_with_preserved(|id| pa.is_preserved_id(id));
                }
                Pass::Function(fp) => {
                    for (_, func) in module.functions.iter_mut() {
                        let pa = fp.run(func, &mut self.am, &self.config);
                        if pa.changed() {
                            changed = true;
                            func.bump_revision();
                        }
                        self.am
                            .invalidate_with_preserved(|id| pa.is_preserved_id(id));
                    }
                }
            }
        }
        changed
    }

    /// 单独在某个函数上运行已注册的操作。
    pub fn run_on_function(&mut self, func: &mut Function) -> bool {
        let mut changed = false;
        for pass in &self.passes {
            match pass {
                Pass::Function(fp) => {
                    let pa = fp.run(func, &mut self.am, &self.config);
                    if pa.changed() {
                        changed = true;
                        func.bump_revision();
                    }
                    self.am
                        .invalidate_with_preserved(|id| pa.is_preserved_id(id));
                }
                Pass::Module(_) => {
                    // ModulePass 无法在单个函数上运行
                }
            }
        }
        changed
    }
}
