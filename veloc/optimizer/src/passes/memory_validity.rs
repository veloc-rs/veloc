//! A successful read proves its byte range readable until a possible lifetime
//! change. It does not prove the bytes unchanged, so this pass only removes
//! unused reads; value forwarding remains a separate memory optimization.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use veloc_analyzer::AnalysisManager;
use veloc_mir::memory::Access;

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
    ) -> PreservedAnalyses {
        let Some(layout) = config.data_layout.as_ref() else {
            return PreservedAnalyses::all();
        };
        let f = am.function_mut();
        let mut dead = Vec::new();
        for block in f.layout().block_order() {
            let mut reads = Vec::<(Access, i64)>::new();
            for inst in f.layout().block_insts(block) {
                let view = f.dfg().inst(inst);
                if view.memory_effect().may_free() || view.has_volatile_access() {
                    reads.clear();
                    continue;
                }
                if view.opcode() != veloc_mir::Opcode::Load {
                    continue;
                }
                let Some(access) = inst.memory_access(f.dfg()) else {
                    continue;
                };
                if access.stored.is_some() {
                    continue;
                }
                let Some(bytes) = access.bytes(layout) else {
                    continue;
                };
                let Some(end) = access.offset.checked_add(i64::from(bytes)) else {
                    continue;
                };
                let covered = reads.iter().any(|&(previous, previous_end)| {
                    previous.ptr == access.ptr
                        && previous.offset <= access.offset
                        && end <= previous_end
                        && previous.flags.alignment() >= access.flags.alignment()
                        && access
                            .offset
                            .checked_sub(previous.offset)
                            .is_some_and(|delta| delta % i64::from(access.flags.alignment()) == 0)
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
                    reads.push((access, end));
                }
            }
        }
        if dead.is_empty() {
            return PreservedAnalyses::all();
        }
        metrics.count("memory.redundant_reads", dead.len() as u64);
        f.edit().erase_insts(&dead);
        PreservedAnalyses::none()
    }
}
