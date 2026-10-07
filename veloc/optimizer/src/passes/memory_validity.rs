//! A successful read proves its byte range readable until a possible lifetime
//! change. It does not prove the bytes unchanged, so this pass only removes
//! unused reads; value forwarding remains a separate memory optimization.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{InstView, MemFlags, memory::Location};

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
        let Some(layout) = config.data_layout.as_ref() else {
            return PassOutcome::Unchanged;
        };
        let f = am.function_mut();
        let mut dead = Vec::new();
        for block in f.layout().block_order() {
            let mut reads = Vec::<(Location, MemFlags, i64)>::new();
            for inst in f.layout().block_insts(block) {
                let view = f.dfg().inst(inst);
                if view.memory_effect().may_free() || view.has_volatile_access() {
                    reads.clear();
                    continue;
                }
                let InstView::Load { flags, .. } = view else {
                    continue;
                };
                let Some(location) = f.memory_location(inst, layout) else {
                    continue;
                };
                let Some(end) = location.offset.checked_add(i64::from(location.bytes)) else {
                    continue;
                };
                let covered = reads
                    .iter()
                    .any(|&(previous, previous_flags, previous_end)| {
                        previous.base == location.base
                            && previous.offset <= location.offset
                            && end <= previous_end
                            && previous_flags.alignment() >= flags.alignment()
                            && location
                                .offset
                                .checked_sub(previous.offset)
                                .is_some_and(|delta| delta % i64::from(flags.alignment()) == 0)
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
                    reads.push((location, flags, end));
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
