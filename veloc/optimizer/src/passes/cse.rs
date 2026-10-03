//! Dominance-scoped commoning after code motion and recurrence construction.
//! The generated instruction view includes every property in equality/hash.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};
use veloc_analyzer::{AnalysisManager, graph::DominatorTree};
use veloc_mir::Inst;

pub struct CsePass;
impl FunctionPass for CsePass {
    fn name(&self) -> &'static str {
        "CsePass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        _: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let f = am.function_mut();
        let dom = DominatorTree::compute(f.cfg(), f.entry_block());
        let blocks = f.cfg().compute_post_order(f.entry_block());
        let mut table = HashMap::<u64, Vec<Inst>>::new();
        let mut changed = 0;
        for block in blocks.into_iter().rev() {
            let insts: Vec<_> = f.layout().block_insts(block).collect();
            for inst in insts {
                let view = f.dfg().inst(inst);
                if !view.can_cse() {
                    continue;
                }
                let types: Vec<_> = f
                    .dfg()
                    .inst_results(inst)
                    .iter()
                    .map(|&v| f.dfg().value_type(v))
                    .collect();
                let mut hash = DefaultHasher::new();
                view.hash(&mut hash);
                types.hash(&mut hash);
                let bucket = table.entry(hash.finish()).or_default();
                let previous = bucket.iter().copied().find(|&old| {
                    dom.dominates(f.layout().inst_block(old).unwrap(), block)
                        && f.dfg().inst(old) == view
                        && f.dfg()
                            .inst_results(old)
                            .iter()
                            .map(|&v| f.dfg().value_type(v))
                            .eq(types.iter().copied())
                });
                if let Some(old) = previous {
                    let values = f.dfg().inst_results(old).to_vec();
                    f.edit().replace_results(inst, &values);
                    changed += 1;
                } else {
                    bucket.push(inst);
                }
            }
        }
        metrics.count("cse.removed", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}
