//! A successful read proves its byte range readable until a possible lifetime
//! change. It does not prove the bytes unchanged, so this pass only removes
//! unused reads; value forwarding remains a separate memory optimization.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{InstView, memory::Access};

pub struct MemoryValidityPass;

impl FunctionPass for MemoryValidityPass {
    fn name(&self) -> &'static str {
        "MemoryValidityPass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        let layout = &config.data_layout;
        let decls = am.decls();
        let f = am.function_mut();
        let mut dead = Vec::new();
        for block in f.layout().block_order() {
            let mut reads = Vec::<Access>::new();
            for inst in f.layout().block_insts(block) {
                let view = f.dfg().inst(inst);
                if view.memory_effect(decls).may_free() || view.has_volatile_access() {
                    reads.clear();
                    continue;
                }
                let InstView::Load { .. } = view else {
                    continue;
                };
                let Some(access) = f.access(inst, layout) else {
                    continue;
                };
                let covered = reads.iter().any(|previous| {
                    previous.covers(access)
                        && previous.flags.alignment() >= access.flags.alignment()
                        && access.offset.abs_diff(previous.offset)
                            % u64::from(access.flags.alignment())
                            == 0
                });
                if covered
                    && f.dfg()
                        .inst_results(inst)
                        .iter()
                        .all(|&v| f.dfg().uses(v).next().is_none())
                {
                    dead.push(inst);
                } else {
                    // Forgetting a proof limits work without changing legality.
                    if reads.len() == 64 {
                        reads.remove(0);
                    }
                    reads.push(access);
                }
            }
        }
        if dead.is_empty() {
            return PassOutcome::Unchanged;
        }
        metrics.count("memory.redundant_reads", dead.len() as u64);
        f.edit().erase_insts(&dead);
        PassOutcome::Changed
    }
}
