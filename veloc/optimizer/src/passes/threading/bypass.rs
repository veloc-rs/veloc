//! Fold small blocks under incoming parameter bindings and branch conditions.
//! Only the incoming edge changes; other predecessors keep the original body.
use crate::path_facts::PathFacts;
use crate::rewrite::{self, Fold, Successors};
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use smallvec::{SmallVec, smallvec};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, Inst, InstView, ScalarConst, SuccessorData, function::EdgeRef};

const MAX_PATH_INSTS: usize = 32;

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
        let mut changed = 0;
        let mut next = f.layout().block_order().next();
        while let Some(block) = next {
            if let Some(last) = f.layout().last_inst(block) {
                changed += bypass(f, block, last);
            }
            next = f.layout().next_block(block);
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
    for (index, dest) in edges.into_iter().enumerate() {
        let edge = EdgeRef {
            inst: last,
            index: index as u32,
        };
        if let Some(target) = trace_edge(f, block, edge, dest) {
            f.edit().redirect_edge(edge, target.block, &target.args);
            changed += 1;
        }
    }
    changed
}

fn trace_edge(
    f: &mut FuncBody,
    source: Block,
    edge: EdgeRef,
    mut dest: SuccessorData,
) -> Option<SuccessorData> {
    let mut bindings = PathFacts::default();
    match f.dfg().inst(edge.inst) {
        InstView::Br { condition, .. } => {
            if !bindings.assume(f, condition, edge.index == 0) {
                return None;
            }
        }
        // The final successor is the default and has no single index value.
        InstView::BrTable { index, table } if (edge.index as usize) < table.len() - 1 => {
            let case = ScalarConst::from(edge.index);
            if let Some(known) = f.dfg().as_scalar_const(index) {
                if known != case {
                    return None;
                }
            } else {
                let literal = f.edit().constant(case.into());
                bindings.bind(index, literal);
            }
        }
        _ => {}
    }

    let original = dest.block;
    let mut seen = HashSet::<Block>::from([source]);
    let mut remaining = MAX_PATH_INSTS;
    while seen.insert(dest.block) {
        let next = fold_successor(f, &dest, &mut bindings, &mut remaining);
        // An exhausted trace may still be inside a cycle. Redirecting its
        // prefix could peel the loop on every CFG round.
        if remaining == 0 {
            return None;
        }
        let Some(next) = next else {
            return (dest.block != original).then_some(dest);
        };
        dest = next;
    }
    None
}

fn fold_successor(
    f: &mut FuncBody,
    edge: &SuccessorData,
    bindings: &mut PathFacts,
    remaining: &mut usize,
) -> Option<SuccessorData> {
    // A redirected edge cannot supply definitions used directly in other
    // blocks. Values passed explicitly as successor arguments can be remapped.
    let escapes = |f: &FuncBody, value| {
        f.dfg()
            .uses(value)
            .any(|site| f.layout().inst_block(site.inst()) != Some(edge.block))
    };
    if f.dfg()
        .block_params(edge.block)
        .iter()
        .any(|&v| escapes(f, v))
    {
        return None;
    }
    for (&param, &arg) in f.dfg().block_params(edge.block).iter().zip(&edge.args) {
        bindings.bind(param, arg);
    }
    let mut cursor = f.layout().first_inst(edge.block);
    while let Some(inst) = cursor {
        *remaining = remaining.checked_sub(1)?;
        let view = f.dfg().inst(inst);
        if view.is_terminator() {
            let chosen = if matches!(view, InstView::Jump { .. }) {
                0
            } else {
                let Successors::One(chosen) =
                    rewrite::successors(view, |v| bindings.fact(f.dfg(), v))
                else {
                    return None;
                };
                chosen
            };
            let dest = view.successor(chosen).unwrap();
            return Some(SuccessorData {
                block: dest.block,
                args: dest.args.iter().map(|&v| bindings.resolve(v)).collect(),
            });
        }

        let results: SmallVec<[_; 2]> = f.dfg().inst_results(inst).into();
        if results.iter().any(|&v| escapes(f, v)) {
            return None;
        }
        let types: SmallVec<[_; 2]> = results.iter().map(|&v| f.dfg().value_type(v)).collect();
        let args: SmallVec<[_; 4]> = f
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| bindings.resolve(v))
            .collect();
        let folds = rewrite::fold(f.dfg().inst_fields(inst), &args, &types, |v| {
            bindings.fact(f.dfg(), v).constant()
        })
        .or_else(|| {
            let InstView::IntCompare { kind, .. } = view else {
                return None;
            };
            let truth = bindings.comparison(f.dfg(), kind, [args[0], args[1]])?;
            Some(smallvec![Fold::Constant(ScalarConst::from(truth))])
        })?;
        // Successful folding proves these operations have no effects or traps
        // under this edge's inputs. Keep their replacements local to the trace.
        for (result, fold) in results.into_iter().zip(folds) {
            let value = match fold {
                Fold::Operand(index) => args[index],
                Fold::Constant(value) => f.edit().constant(value.into()),
            };
            bindings.bind(result, value);
        }
        cursor = f.layout().next_inst(inst);
    }
    None
}
