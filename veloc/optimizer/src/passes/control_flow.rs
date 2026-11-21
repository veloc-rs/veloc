//! Iterate branch threading and structural cleanup at pipeline boundaries.
//! Speculative condition merging runs separately before expression optimization.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use veloc_analyzer::AnalysisManager;

pub struct ControlFlowPass;
impl FunctionPass for ControlFlowPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "ControlFlowPass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        use super::{cfg::CfgPass, threading::BypassPass};
        let mut changed = am.function_mut().edit().remove_unreachable();
        loop {
            let mut round = false;
            // Keep specialization before merging: merging can hide a small
            // comparison/dispatch block that could have been bypassed.
            for pass in [&BypassPass as &dyn FunctionPass, &CfgPass] {
                let scope = metrics.scope(pass.name(), 0);
                round |= pass.run(am, config, metrics).changed();
                scope.success();
            }
            changed |= round;
            if !round {
                break;
            }
        }
        if changed {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}
