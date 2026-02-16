use crate::liveness::{Liveness, analyze_liveness};
use crate::use_def::UseDefAnalysis;
use core::any::TypeId;
use veloc_ir::function::Function;

/// 分析管理器：负责计算和缓存分析结果。
#[derive(Default)]
pub struct AnalysisManager {
    /// 函数级的 Use-Def 分析侧表缓存。
    use_def: Option<(u64, UseDefAnalysis)>,
    /// 活跃变量分析。
    liveness: Option<(u64, Liveness)>,
}

impl AnalysisManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取函数的 Use-Def 分析结果。
    pub fn use_def(&mut self, func: &Function) -> &UseDefAnalysis {
        let rev = func.revision();
        // 如果版本不对就重算
        if self
            .use_def
            .as_ref()
            .map(|(r, _)| *r != rev)
            .unwrap_or(true)
        {
            self.use_def = Some((rev, UseDefAnalysis::new(func)));
        }
        &self.use_def.as_ref().unwrap().1
    }

    /// 获取函数的活跃变量分析结果。
    pub fn liveness(&mut self, func: &Function) -> &Liveness {
        let rev = func.revision();
        if self
            .liveness
            .as_ref()
            .map(|(r, _)| *r != rev)
            .unwrap_or(true)
        {
            self.liveness = Some((rev, analyze_liveness(func)));
        }
        &self.liveness.as_ref().unwrap().1
    }

    /// 根据保留通知来使特定分析失效。
    pub fn invalidate_with_preserved<F>(&mut self, checker: F)
    where
        F: Fn(TypeId) -> bool,
    {
        if !checker(TypeId::of::<UseDefAnalysis>()) {
            self.use_def = None;
        }
        if !checker(TypeId::of::<Liveness>()) {
            self.liveness = None;
        }
    }

    /// 标记分析侧表失效。
    pub fn invalidate(&mut self) {
        self.use_def = None;
        self.liveness = None;
    }
}
