//! Bypass dispatch blocks using facts specific to each incoming edge.
//! This does not clone instructions or specialize general block bodies.
use crate::rewrite::{self, Fact, Successors};
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, Inst, InstView, SuccessorData, function::EdgeRef};
pub struct BypassPass;
impl FunctionPass for BypassPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "BypassPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let blocks: Vec<_> = f.layout().block_order().collect();
        let mut changed = 0;
        for block in blocks {
            let Some(last) = f.layout().last_inst(block) else {
                continue;
            };
            changed += bypass(f, block, last);
        }
        metrics.count("threading.bypassed", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}
fn bypass(f: &mut FuncBody, block: Block, last: Inst) -> u64 {
    let mut changed = 0;
    let mut edges = Vec::new();
    f.dfg()
        .inst(last)
        .visit_successors(|e| edges.push(SuccessorData::new(e.block, e.args)));
    for (index, mut edge) in edges.into_iter().enumerate() {
        let original = edge.block;
        let mut seen = HashSet::from([block]);
        while seen.insert(edge.block) {
            if let Some(next) = thread_comparison(f, last, index, &edge) {
                if seen.contains(&next.block) {
                    break;
                }
                edge = next;
                continue;
            }
            if let Some(next) = crate::passes::cfg::forward_successor(f, &edge) {
                if seen.contains(&next.block) {
                    break;
                }
                edge = next;
                continue;
            }
            let Some(inst) = f.layout().first_inst(edge.block) else {
                break;
            };
            if f.layout().last_inst(edge.block) != Some(inst) {
                break;
            }
            let params = f.dfg().block_params(edge.block);
            let view = f.dfg().inst(inst);
            let Successors::One(chosen) = rewrite::successors(view, |value| {
                let mapped = params
                    .iter()
                    .position(|&p| p == value)
                    .map_or(value, |i| edge.args[i]);
                f.dfg()
                    .as_scalar_const(mapped)
                    .map_or(Fact::Varying, Fact::Constant)
            }) else {
                break;
            };
            let dest = view.successor(chosen).expect("selected successor");
            if f.dfg()
                .block_params(edge.block)
                .iter()
                .any(|&p| f.dfg().uses(p).any(|site| site.inst() != inst))
            {
                break;
            }
            if seen.contains(&dest.block) {
                break;
            }
            let params = f.dfg().block_params(edge.block);
            let args = dest
                .args
                .iter()
                .map(|v| {
                    params
                        .iter()
                        .position(|p| p == v)
                        .map_or(*v, |i| edge.args[i])
                })
                .collect::<Vec<_>>();
            edge = SuccessorData::new(dest.block, &args);
        }
        if edge.block != original {
            f.edit().redirect_edge(
                EdgeRef {
                    inst: last,
                    index: index as u32,
                },
                edge.block,
                &edge.args,
            );
            changed += 1;
        }
    }
    changed
}
/// Bypass a comparison-only block when the incoming branch already proves
/// its outcome. Relational outcome sets avoid ad hoc signed/unsigned cases.
fn thread_comparison(
    f: &FuncBody,
    source: Inst,
    edge_index: usize,
    edge: &SuccessorData,
) -> Option<SuccessorData> {
    let InstView::Br { condition, .. } = f.dfg().inst(source) else {
        return None;
    };
    let source_cmp = f.dfg().value_inst(condition)?;
    let InstView::IntCompare {
        kind: source_kind,
        args: source_args,
    } = f.dfg().inst(source_cmp)
    else {
        return None;
    };
    let insts: Vec<_> = f.layout().block_insts(edge.block).collect();
    let [compare, branch] = insts.as_slice() else {
        return None;
    };
    let InstView::IntCompare { kind, args } = f.dfg().inst(*compare) else {
        return None;
    };
    let result = f.dfg().first_result(*compare)?;
    let InstView::Br {
        condition,
        then_dest,
        else_dest,
    } = f.dfg().inst(*branch)
    else {
        return None;
    };
    if condition != result || f.dfg().uses(result).any(|u| u.inst() != *branch) {
        return None;
    }
    let params = f.dfg().block_params(edge.block);
    if params.iter().any(|&v| {
        f.dfg()
            .uses(v)
            .any(|u| u.inst() != *compare && u.inst() != *branch)
    }) {
        return None;
    }
    let mapped = |v| {
        params
            .iter()
            .position(|&p| p == v)
            .map_or(v, |i| edge.args[i])
    };
    let args = [mapped(args[0]), mapped(args[1])];
    let kind = if args == [source_args[1], source_args[0]] {
        kind.swap()
    } else if args == *source_args {
        kind
    } else {
        return None;
    };
    let known = if edge_index == 1 {
        source_kind.complement()
    } else {
        source_kind
    };
    let dest = if known.implies(kind)? {
        then_dest
    } else {
        else_dest
    };
    // The comparison's result cannot become an argument on the new edge.
    if dest.args.contains(&result) {
        return None;
    }
    let args: Vec<_> = dest.args.iter().map(|&v| mapped(v)).collect();
    Some(SuccessorData::new(dest.block, &args))
}
