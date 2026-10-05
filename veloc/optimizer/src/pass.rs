use crate::{Error, Profile, Result};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::Module;

/// Whether a pass changed IR. Analysis invalidation belongs to the editor,
/// independently of whether a pass ultimately reports a change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassOutcome {
    Unchanged,
    Changed,
}

impl PassOutcome {
    pub fn changed(self) -> bool {
        self == Self::Changed
    }
}

/// 作用于单个函数的优化 Pass。
pub trait FunctionPass {
    fn name(&self) -> &'static str;
    /// Opt in only when instances sharing this key have identical behavior for
    /// the same IR and OptConfig. The manager caches an unchanged result until
    /// a pass reports a change or a module-pass boundary; changed runs are never cached.
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        None
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome;
}

/// 作用于整个模块的优化 Pass。
pub trait ModulePass {
    fn name(&self) -> &'static str;
    fn run(&self, module: &mut Module, config: &OptConfig, metrics: &Profile) -> PassOutcome;
}

/// 优化配置。
#[derive(Debug, Clone, Default)]
pub struct OptConfig {
    /// None disables optimizations that depend on target memory representation.
    pub data_layout: Option<veloc_types::DataLayout>,
    /// Optional learned profitability decisions; passes still enforce legality.
    pub policy: Option<std::sync::Arc<veloc_policy::Policy>>,
    /// 调试标签系统，用于控制细粒度的输出，如 "dce", "liveness" 等
    debug_tags: HashSet<String>,
}

impl OptConfig {
    /// 创建配置并批量添加调试标签
    /// 如果发现未知标签，直接返回错误
    pub fn with_debug_tags(tags: &[&str]) -> Result<Self> {
        let mut config = Self::default();
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
