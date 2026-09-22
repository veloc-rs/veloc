use crate::liveness::{Liveness, analyze_liveness};
use core::any::TypeId;
use veloc_mir::FuncBody;

/// Analyses belong to one borrowed function. Exclusive mutation conservatively
/// invalidates caches, without revision counters or cross-function reuse.
pub struct AnalysisManager<'f> {
    func: &'f mut FuncBody,
    liveness: Option<Liveness>,
}

impl<'f> AnalysisManager<'f> {
    pub fn new(func: &'f mut FuncBody) -> Self {
        Self {
            func,
            liveness: None,
        }
    }

    pub fn function(&self) -> &FuncBody {
        self.func
    }

    pub fn function_mut(&mut self) -> &mut FuncBody {
        self.invalidate();
        self.func
    }

    pub fn liveness(&mut self) -> &Liveness {
        self.liveness
            .get_or_insert_with(|| analyze_liveness(self.func))
    }

    pub fn invalidate_with_preserved(&mut self, checker: impl Fn(TypeId) -> bool) {
        if !checker(TypeId::of::<Liveness>()) {
            self.liveness = None;
        }
    }

    pub fn invalidate(&mut self) {
        self.liveness = None;
    }
}
