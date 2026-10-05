use crate::Dominators;
use crate::liveness::{Liveness, analyze_liveness};
use veloc_mir::FuncBody;

/// Analyses belong to one borrowed function. Exclusive mutation conservatively
/// invalidates caches, without revision counters or cross-function reuse.
pub struct AnalysisManager<'f> {
    func: &'f mut FuncBody,
    #[cfg(feature = "profile")]
    profile: veloc_profile::Profile,
    liveness: Option<Liveness>,
    dominators: Option<Dominators>,
}

impl<'f> AnalysisManager<'f> {
    pub fn new(func: &'f mut FuncBody) -> Self {
        Self {
            func,
            #[cfg(feature = "profile")]
            profile: veloc_profile::Profile::default(),
            liveness: None,
            dominators: None,
        }
    }

    #[cfg(feature = "profile")]
    pub fn with_profile(mut self, profile: veloc_profile::Profile) -> Self {
        self.profile = profile;
        self
    }

    pub fn function(&self) -> &FuncBody {
        self.func
    }

    pub fn function_mut(&mut self) -> &mut FuncBody {
        self.invalidate();
        self.func
    }

    pub fn liveness(&mut self) -> &Liveness {
        #[cfg(feature = "profile")]
        if self.liveness.is_some() {
            self.profile.count("analysis.liveness.cache_hits", 1);
        }
        self.liveness.get_or_insert_with(|| {
            #[cfg(feature = "profile")]
            let scope = self.profile.scope("analysis.liveness", 0);
            let result = analyze_liveness(self.func);
            #[cfg(feature = "profile")]
            scope.success();
            result
        })
    }

    pub fn dominators(&mut self) -> &Dominators {
        #[cfg(feature = "profile")]
        if self.dominators.is_some() {
            self.profile.count("analysis.dominators.cache_hits", 1);
        }
        self.dominators.get_or_insert_with(|| {
            #[cfg(feature = "profile")]
            let scope = self.profile.scope("analysis.dominators", 0);
            let result = Dominators::compute(self.func.cfg(), self.func.entry_block());
            #[cfg(feature = "profile")]
            scope.success();
            result
        })
    }

    /// Move the snapshot into a transforming pass without cloning its tables.
    /// The pass must not use it after changing CFG topology.
    pub fn take_dominators(&mut self) -> Dominators {
        self.dominators();
        self.dominators.take().unwrap()
    }

    pub fn invalidate(&mut self) {
        self.liveness = None;
        self.dominators = None;
    }
}
