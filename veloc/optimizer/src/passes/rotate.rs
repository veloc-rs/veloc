//! Rotate simple guarded loops without assuming a nonzero trip count.
//! The first header remains the entry guard. Its values become body/exit
//! parameters, so the latch can test the next iteration without an extra jump.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{HashMap, HashSet};
use veloc_analyzer::{
    AnalysisManager,
    graph::{DominatorTree, LoopInfo},
};
use veloc_mir::{InstView, Value, function::EdgeRef};

pub struct RotatePass;
impl FunctionPass for RotatePass {
    fn name(&self) -> &'static str {
        "RotatePass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let dom = DominatorTree::compute(f.cfg(), f.entry_block());
        let info = LoopInfo::compute(f.cfg(), &dom);
        let mut changed = 0;
        let mut headers = HashSet::new();
        for &(_, header) in info.backedges() {
            headers.insert(header);
        }
        let mut headers: Vec<_> = headers.into_iter().collect();
        headers.sort();
        for header in headers {
            let dom = DominatorTree::compute(f.cfg(), f.entry_block());
            let latches: Vec<_> = f
                .cfg()
                .preds(header)
                .iter()
                .copied()
                .filter(|&b| dom.dominates(header, b))
                .collect();
            let [latch] = latches[..] else {
                continue;
            };
            let header_insts: Vec<_> = f.layout().block_insts(header).collect();
            let Some((&branch, computations)) = header_insts.split_last() else {
                continue;
            };
            if computations.len() > 4
                || !computations
                    .iter()
                    .all(|&i| f.dfg().inst(i).can_speculate())
            {
                continue;
            }
            let InstView::Br {
                condition,
                then_dest,
                else_dest,
            } = f.dfg().inst(branch)
            else {
                continue;
            };
            let (condition, yes, no) = (
                condition,
                veloc_mir::SuccessorData::new(then_dest.block, then_dest.args),
                veloc_mir::SuccessorData::new(else_dest.block, else_dest.args),
            );
            let mut members = HashSet::from([header]);
            let mut pending = vec![latch];
            while let Some(b) = pending.pop() {
                if members.insert(b) {
                    pending.extend_from_slice(f.cfg().preds(b));
                }
            }
            let (body, exit) = if members.contains(&yes.block) && !members.contains(&no.block) {
                (yes.block, no.block)
            } else if members.contains(&no.block) && !members.contains(&yes.block) {
                (no.block, yes.block)
            } else {
                continue;
            };
            if body == header
                || f.cfg().preds(body) != [header]
                || f.cfg().preds(exit) != [header]
                || members
                    .iter()
                    .any(|&b| b != header && f.cfg().succs(b).iter().any(|s| !members.contains(s)))
            {
                continue;
            }
            let latch_end = f.layout().last_inst(latch).unwrap();
            if !matches!(f.dfg().inst(latch_end),InstView::Jump{dest} if dest.block==header) {
                continue;
            }
            let mut carried = f.dfg().block_params(header).to_vec();
            for &inst in computations {
                carried.extend_from_slice(f.dfg().inst_results(inst));
            }
            let original_uses: Vec<_> = carried
                .iter()
                .map(|&v| {
                    (
                        v,
                        f.dfg()
                            .uses(v)
                            .map(|site| (site.inst(), site.index()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let mut body_values = HashMap::new();
            let mut exit_values = HashMap::new();
            for &value in &carried {
                let ty = f.dfg().value_type(value);
                body_values.insert(value, f.edit().append_block_param(body, ty));
                exit_values.insert(value, f.edit().append_block_param(exit, ty));
            }
            for (value, uses) in original_uses {
                for (inst, index) in uses {
                    let block = f.layout().inst_block(inst).unwrap();
                    if block == header {
                        continue;
                    }
                    let replacement = if members.contains(&block) {
                        body_values[&value]
                    } else {
                        exit_values[&value]
                    };
                    f.edit().set_operand(inst, index, replacement);
                }
            }
            // Read the backedge after replacing loop uses by the body's values.
            let InstView::Jump { dest } = f.dfg().inst(latch_end) else {
                unreachable!()
            };
            let mut next: HashMap<Value, Value> = f
                .dfg()
                .block_params(header)
                .iter()
                .copied()
                .zip(dest.args.iter().copied())
                .collect();
            for &inst in computations {
                let args: Vec<_> = f
                    .dfg()
                    .operands(inst)
                    .iter()
                    .map(|v| next.get(v).copied().unwrap_or(*v))
                    .collect();
                let old = f.dfg().inst_results(inst).to_vec();
                let types: Vec<_> = old.iter().map(|&v| f.dfg().value_type(v)).collect();
                let new = f.edit().insert_before(
                    latch_end,
                    |w| w.copy_with_operands(inst, &args),
                    &types,
                );
                for (old, &new) in old.into_iter().zip(f.dfg().inst_results(new)) {
                    next.insert(old, new);
                }
            }
            let remap = |edge: &veloc_mir::SuccessorData, values: &HashMap<Value, Value>| {
                let mut args: Vec<_> = edge
                    .args
                    .iter()
                    .map(|v| values.get(v).copied().unwrap_or(*v))
                    .collect();
                args.extend(carried.iter().map(|v| values.get(v).copied().unwrap_or(*v)));
                veloc_mir::SuccessorData::new(edge.block, &args)
            };
            let first_yes = remap(&yes, &HashMap::new());
            let first_no = remap(&no, &HashMap::new());
            f.edit().redirect_edge(
                EdgeRef {
                    inst: branch,
                    index: 0,
                },
                yes.block,
                &first_yes.args,
            );
            f.edit().redirect_edge(
                EdgeRef {
                    inst: branch,
                    index: 1,
                },
                no.block,
                &first_no.args,
            );
            let next_yes = remap(&yes, &next);
            let next_no = remap(&no, &next);
            let condition = next.get(&condition).copied().unwrap_or(condition);
            f.edit().replace_inst(latch_end, |w| {
                w.br(condition, next_yes.as_view(), next_no.as_view())
            });
            changed += 1;
        }
        metrics.count("rotate.loops", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}
