//! Hoist speculatable, loop-invariant computations into existing preheaders.
//! Mutable loads and trapping operations require additional proofs and stay put.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{BTreeMap, BTreeSet};
use veloc_analyzer::{
    AnalysisManager,
    graph::{DominatorTree, LoopInfo},
};
use veloc_mir::{Block, SuccessorData, ValueDef, function::EdgeRef};

pub struct LicmPass;

impl FunctionPass for LicmPass {
    fn name(&self) -> &'static str {
        "LicmPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let dom = DominatorTree::compute(f.cfg(), f.entry_block());
        let info = LoopInfo::compute(f.cfg(), &dom);
        let mut loops = BTreeMap::<Block, BTreeSet<Block>>::new();
        for &(latch, header) in info.backedges() {
            let members = loops.entry(header).or_default();
            members.insert(header);
            let mut pending = vec![latch];
            while let Some(block) = pending.pop() {
                if members.insert(block) {
                    pending.extend_from_slice(f.cfg().preds(block));
                }
            }
        }
        let mut loops: Vec<_> = loops.into_iter().collect();
        loops.sort_by_key(|(_, members)| members.len());
        let mut changed = 0;
        for (header, mut members) in loops {
            // Earlier inner-loop edits may have introduced new preheaders.
            let dom = DominatorTree::compute(f.cfg(), f.entry_block());
            members.clear();
            members.insert(header);
            let mut pending: Vec<_> = f
                .cfg()
                .preds(header)
                .iter()
                .copied()
                .filter(|&b| dom.dominates(header, b))
                .collect();
            while let Some(b) = pending.pop() {
                if members.insert(b) {
                    pending.extend_from_slice(f.cfg().preds(b));
                }
            }
            let entries: Vec<_> = f
                .cfg()
                .preds(header)
                .iter()
                .copied()
                .filter(|p| !members.contains(p))
                .collect();
            let [mut preheader] = entries[..] else {
                continue;
            };
            if f.cfg().succs(preheader) != [header] {
                let branch = f.layout().last_inst(preheader).unwrap();
                let mut incoming = Vec::new();
                let mut index = 0;
                f.dfg().inst(branch).visit_successors(|edge| {
                    if edge.block == header {
                        incoming.push((index, SuccessorData::new(header, edge.args)));
                    }
                    index += 1;
                });
                let [(index, edge)] = &incoming[..] else {
                    continue;
                };
                let new = f.edit().create_block();
                f.edit().append_block(new);
                f.edit().append_inst(new, |w| w.jump(edge.as_view()), &[]);
                f.edit().redirect_edge(
                    EdgeRef {
                        inst: branch,
                        index: *index,
                    },
                    new,
                    &[],
                );
                preheader = new;
                changed += 1;
            }
            let dom = DominatorTree::compute(f.cfg(), f.entry_block());
            let anchor = f.layout().last_inst(preheader).unwrap();
            let mut candidates: Vec<_> = f
                .layout()
                .block_order()
                .filter(|b| members.contains(b))
                .flat_map(|b| f.layout().block_insts(b))
                .filter(|&i| f.dfg().inst(i).can_speculate())
                .collect();
            loop {
                let before = candidates.len();
                candidates.retain(|&inst| {
                    let invariant =
                        f.dfg()
                            .operands(inst)
                            .iter()
                            .all(|&v| match f.dfg().value_def(v) {
                                ValueDef::FunctionParam(_) | ValueDef::Const(_) => true,
                                ValueDef::BlockParam(b) => {
                                    !members.contains(&b) && dom.dominates(b, preheader)
                                }
                                ValueDef::Inst(i) => {
                                    let b = f.layout().inst_block(i).unwrap();
                                    !members.contains(&b) && dom.dominates(b, preheader)
                                }
                            });
                    if invariant {
                        f.edit().move_before(inst, anchor);
                        changed += 1;
                    }
                    !invariant
                });
                if before == candidates.len() {
                    break;
                }
            }
        }
        metrics.count("licm.hoisted", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}
