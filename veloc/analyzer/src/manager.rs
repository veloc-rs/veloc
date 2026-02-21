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

    /// 获取函数的 Use-Def 分析结果（可变引用）。
    /// 调用者如果通过此引用在 DFG 中替换了值，应调用 `update_use_def_revision` 同步版本号，
    /// 以避免 AnalysisManager 认为数据过期而触发不必要的重算。
    pub fn use_def_mut(&mut self, func: &Function) -> &mut UseDefAnalysis {
        let rev = func.revision();
        if self
            .use_def
            .as_ref()
            .map(|(r, _)| *r != rev)
            .unwrap_or(true)
        {
            self.use_def = Some((rev, UseDefAnalysis::new(func)));
        }
        &mut self.use_def.as_mut().unwrap().1
    }

    /// 手动更新缓存的 Use-Def 分析版本号。
    pub fn update_use_def_revision(&mut self, func: &Function) {
        if let Some((r, _)) = &mut self.use_def {
            *r = func.revision();
        }
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
