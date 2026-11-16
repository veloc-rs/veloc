//! Structural CFG cleanup: local rewrites, forwarding blocks and block merges.
//! No abstract interpretation, path predicates or speculative execution.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{HashSet, VecDeque};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, Inst, InstView, SuccessorData, ValueDef, function::EdgeRef};

pub struct CfgPass;

#[derive(Default)]
struct Worklist {
    queue: VecDeque<Block>,
    queued: HashSet<Block>,
}
impl Worklist {
    fn push(&mut self, block: Block) {
        if self.queued.insert(block) {
            self.queue.push_back(block);
        }
    }
    fn pop(&mut self) -> Option<Block> {
        let block = self.queue.pop_front()?;
        self.queued.remove(&block);
        Some(block)
    }
    fn neighborhood(&mut self, f: &FuncBody, block: Block) {
        self.push(block);
        for &pred in f.cfg().preds(block) {
            self.push(pred);
        }
        for &succ in f.cfg().succs(block) {
            self.push(succ);
            // Losing an incoming edge can enable another predecessor to merge.
            for &pred in f.cfg().preds(succ) {
                self.push(pred);
            }
        }
    }
}

impl FunctionPass for CfgPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "CfgPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let mut changed = u64::from(f.edit().remove_unreachable());
        let mut work = Worklist::default();
        for block in f.layout().block_order() {
            work.push(block);
        }
        let mut visits = 0;
        loop {
            while let Some(block) = work.pop() {
                if !f.layout().contains_block(block) {
                    continue;
                }
                visits += 1;
                let Some(last) = f.layout().last_inst(block) else {
                    continue;
                };
                let old_successors = f.cfg().succs(block).to_vec();
                let mut parameter_blocks: Vec<_> = f
                    .dfg()
                    .operands(last)
                    .iter()
                    .filter_map(|&v| match f.dfg().value_def(v) {
                        ValueDef::BlockParam(block) => Some(block),
                        _ => None,
                    })
                    .collect();
                let mut edited = false;
                if let Some(rewrite) = crate::rewrite::find_rewrite(f, last) {
                    rewrite.apply(f, last);
                    edited = true;
                }
                edited |= forward_edges(f, block, last);
                // Merging substitutes block parameters, including uses in blocks
                // other than the merged pair. Revisit those users as well.
                loop {
                    let tail = f.layout().last_inst(block).expect("terminated block");
                    let users = merge_users(f, tail);
                    let definitions: Vec<_> = f
                        .dfg()
                        .operands(tail)
                        .iter()
                        .filter_map(|&v| match f.dfg().value_def(v) {
                            ValueDef::BlockParam(block) => Some(block),
                            _ => None,
                        })
                        .collect();
                    if !f.edit().merge_successor(block) {
                        break;
                    }
                    edited = true;
                    parameter_blocks.extend(definitions);
                    for user in users {
                        work.push(user);
                    }
                }
                if edited {
                    changed += 1;
                    work.neighborhood(f, block);
                    // Removing an escaping parameter use can make its defining
                    // block eligible for forwarding, even outside this neighborhood.
                    for defining in parameter_blocks {
                        if f.layout().contains_block(defining) {
                            work.neighborhood(f, defining);
                        }
                    }
                    for successor in old_successors {
                        if f.layout().contains_block(successor) {
                            work.neighborhood(f, successor);
                        }
                    }
                }
            }
            // Reachability remains a phase-level graph walk. Deleting blocks
            // can remove escaping SSA uses anywhere, so reseed after deletion.
            if !f.edit().remove_unreachable() {
                break;
            }
            changed += 1;
            for block in f.layout().block_order() {
                work.push(block);
            }
        }
        metrics.count("cfg.simplified", changed);
        metrics.count("cfg.block_visits", visits);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

fn forward_edges(f: &mut FuncBody, block: Block, inst: Inst) -> bool {
    let mut edges = Vec::new();
    f.dfg()
        .inst(inst)
        .visit_successors(|e| edges.push(SuccessorData::new(e.block, e.args)));
    let mut changed = false;
    for (index, mut edge) in edges.into_iter().enumerate() {
        let original = edge.block;
        let mut seen = HashSet::from([block]);
        while seen.insert(edge.block) {
            let Some(next) = forward_successor(f, &edge) else {
                break;
            };
            if seen.contains(&next.block) {
                break;
            }
            edge = next;
        }
        if edge.block != original {
            f.edit().redirect_edge(
                EdgeRef {
                    inst,
                    index: index as u32,
                },
                edge.block,
                &edge.args,
            );
            changed = true;
        }
    }
    changed
}

/// Map an incoming edge through a jump-only block without leaving parameter uses behind.
fn forward_successor(f: &FuncBody, edge: &SuccessorData) -> Option<SuccessorData> {
    let only = f.layout().first_inst(edge.block)?;
    if f.layout().last_inst(edge.block) != Some(only) {
        return None;
    }
    let InstView::Jump { dest } = f.dfg().inst(only) else {
        return None;
    };
    let params = f.dfg().block_params(edge.block);
    if params
        .iter()
        .any(|&p| f.dfg().uses(p).any(|u| u.inst() != only))
    {
        return None;
    }
    let args: Vec<_> = dest
        .args
        .iter()
        .map(|v| {
            params
                .iter()
                .position(|p| p == v)
                .map_or(*v, |i| edge.args[i])
        })
        .collect();
    Some(SuccessorData::new(dest.block, &args))
}

fn merge_users(f: &FuncBody, inst: Inst) -> Vec<Block> {
    let InstView::Jump { dest } = f.dfg().inst(inst) else {
        return Vec::new();
    };
    f.dfg()
        .block_params(dest.block)
        .iter()
        .flat_map(|&param| {
            f.dfg()
                .uses(param)
                .filter_map(|site| f.layout().inst_block(site.inst()))
        })
        .collect()
}
