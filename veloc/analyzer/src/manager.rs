use crate::Dominators;
use crate::liveness::{Liveness, analyze_liveness};
use core::any::TypeId;
use veloc_mir::FuncBody;

/// Analyses belong to one borrowed function. Exclusive mutation conservatively
/// invalidates caches, without revision counters or cross-function reuse.
pub struct AnalysisManager<'f> {
    func: &'f mut FuncBody,
    liveness: Option<Liveness>,
    dominators: Option<Dominators>,
}

impl<'f> AnalysisManager<'f> {
    pub fn new(func: &'f mut FuncBody) -> Self {
        Self {
            func,
            liveness: None,
            dominators: None,
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

    pub fn dominators(&mut self) -> &Dominators {
        self.dominators
            .get_or_insert_with(|| Dominators::compute(self.func.cfg(), self.func.entry_block()))
    }

    /// Move the snapshot into a transforming pass without cloning its tables.
    /// The pass must not use it after changing CFG topology.
    pub fn take_dominators(&mut self) -> Dominators {
        self.dominators
            .take()
            .unwrap_or_else(|| Dominators::compute(self.func.cfg(), self.func.entry_block()))
    }

    pub fn invalidate_with_preserved(&mut self, checker: impl Fn(TypeId) -> bool) {
        if !checker(TypeId::of::<Dominators>()) {
            self.dominators = None;
        }
        if !checker(TypeId::of::<Liveness>()) {
            self.liveness = None;
        }
    }

    pub fn invalidate(&mut self) {
        self.liveness = None;
        self.dominators = None;
    }
}
