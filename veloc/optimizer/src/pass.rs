use crate::{Error, Result, metrics::Metrics};
use alloc::boxed::Box;
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
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses;
}

/// 作用于整个模块的优化 Pass。
pub trait ModulePass {
    fn name(&self) -> &str;
    fn run(
        &self,
        module: &mut ModuleData,
        am: &mut AnalysisManager,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses;
}

/// 优化 Pass 的类型包装。
pub enum Pass {
    Function(Box<dyn FunctionPass>),
    Module(Box<dyn ModulePass>),
}

impl Pass {
    pub fn name(&self) -> &str {
        match self {
            Pass::Function(f) => f.name(),
            Pass::Module(m) => m.name(),
        }
    }
}

/// 优化配置。
#[derive(Debug, Clone, Default)]
pub struct OptConfig {
    pub monitor_performance: bool,
    /// 调试标签系统，用于控制细粒度的输出，如 "dce", "liveness" 等
    debug_tags: HashSet<String>,
}

impl OptConfig {
    /// 创建基础配置
    pub fn new(monitor_performance: bool) -> Self {
        Self {
            monitor_performance,
            debug_tags: HashSet::new(),
        }
    }

    /// 创建配置并批量添加调试标签
    /// 如果发现未知标签，直接返回错误
    pub fn with_debug_tags(monitor_performance: bool, tags: &[&str]) -> Result<Self> {
        let mut config = Self::new(monitor_performance);
        for tag in tags {
            config.add_debug_tag(tag)?;
        }
        Ok(config)
    }

    /// 检查指定的调试标签是否已启用
    pub fn is_debug_enabled(&self, tag: &str) -> bool {
        self.debug_tags.contains("all") || self.debug_tags.contains(tag)
    }

    /// 添加单个调试标签，如果是未知标签则返回错误
    pub fn add_debug_tag(&mut self, tag: &str) -> Result<()> {
        let known_tags = crate::get_known_debug_tags();
        if tag != "all" && !known_tags.contains(&tag) {
            return Err(Error::UnknownDebugTag(tag.to_string()));
        }
        self.debug_tags.insert(tag.to_string());
        Ok(())
    }
}
