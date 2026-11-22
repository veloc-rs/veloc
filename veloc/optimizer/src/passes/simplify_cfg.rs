//! Simplify control flow by rewriting branches, forwarding edges, and merging blocks.
//! Path-specific folds redirect only the edge being traced.
use crate::path_facts::PathFacts;
use crate::rewrite::{self, Fold, Successors};
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use smallvec::{SmallVec, smallvec};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{
    Block, FuncBody, FuncDecls, Inst, InstView, ScalarConst, SuccessorData, function::EdgeRef,
};

pub struct SimplifyCfgPass;
impl FunctionPass for SimplifyCfgPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "SimplifyCfgPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let decls = am.decls();
        let f = am.function_mut();
        let mut changed = f.edit().remove_unreachable();
        loop {
            // Merge after tracing: a merge can hide a block that one edge could bypass.
            let edges_changed = rewrite_edges(f, decls, metrics);
            let cleaned = cleanup(f, edges_changed, metrics);
            if !edges_changed && !cleaned {
                break;
            }
            changed = true;
        }
        if changed {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}

fn rewrite_edges(f: &mut FuncBody, decls: &FuncDecls, metrics: &Profile) -> bool {
    let scope = metrics.scope("threading.edges", 0);
    let mut rewritten = 0;
    let mut bypassed = 0;
    let mut next = f.layout().block_order().next();
    while let Some(block) = next {
        if let Some(last) = f.layout().last_inst(block) {
            // Rewriting can swap or remove successors. Trace their final order.
            while let Some(edit) = rewrite::find_rewrite(f, last) {
                edit.apply(f, last);
                rewritten += 1;
            }
            bypassed += bypass(f, decls, block, last);
        }
        next = f.layout().next_block(block);
    }
    metrics.count("threading.rewritten", rewritten);
    metrics.count("threading.bypassed", bypassed);
    scope.success();
    rewritten != 0 || bypassed != 0
}

fn cleanup(f: &mut FuncBody, edges_changed: bool, metrics: &Profile) -> bool {
    let scope = metrics.scope("cfg.cleanup", 0);
    // All blocks were reachable before this round; merging alone cannot orphan one.
    let mut edits = u64::from(edges_changed && f.edit().remove_unreachable());
    let mut visits = 0;
    let mut next = f.layout().block_order().next();
    while let Some(block) = next {
        visits += 1;
        while f.edit().merge_successor(block) {
            edits += 1;
        }
        next = f.layout().next_block(block);
    }
    metrics.count("cfg.simplified", edits);
    metrics.count("cfg.block_visits", visits);
    scope.success();
    edits != 0
}

fn bypass(f: &mut FuncBody, decls: &FuncDecls, block: Block, last: Inst) -> u64 {
    let mut changed = 0;
    let mut edges = SmallVec::<[_; 2]>::new();
    f.dfg()
        .inst(last)
        .visit_successors(|e| edges.push(SuccessorData::new(e.block, e.args)));
    for (index, dest) in edges.into_iter().enumerate() {
        let edge = EdgeRef {
            inst: last,
            index: index as u32,
        };
        if let Some(target) = trace_edge(f, decls, block, edge, dest) {
            f.edit().redirect_edge(edge, target.block, &target.args);
            changed += 1;
        }
    }
    changed
}

fn trace_edge(
    f: &mut FuncBody,
    decls: &FuncDecls,
    source: Block,
    edge: EdgeRef,
    mut dest: SuccessorData,
) -> Option<SuccessorData> {
    let mut facts = edge_facts(f, edge)?;
    let original = dest.block;
    let mut seen = HashSet::<Block>::from([source]);
    // Discard a cyclic trace rather than redirecting partway around a loop.
    while seen.insert(dest.block) {
        let Some(next) = fold_successor(f, decls, &dest, &mut facts) else {
            return (dest.block != original).then_some(dest);
        };
        dest = next;
    }
    None
}

fn edge_facts(f: &mut FuncBody, edge: EdgeRef) -> Option<PathFacts> {
    let mut facts = PathFacts::default();
    match f.dfg().inst(edge.inst) {
        InstView::Br { condition, .. } => {
            if !facts.assume(f, condition, edge.index == 0) {
                return None;
            }
        }
        // The final successor is the default and has no single index value.
        InstView::BrTable { index, table } if (edge.index as usize) < table.len() - 1 => {
            let case = ScalarConst::from(edge.index);
            let literal = f.edit().constant(case.into());
            facts.bind(index, literal);
        }
        _ => {}
    }
    Some(facts)
}

fn fold_successor(
    f: &mut FuncBody,
    decls: &FuncDecls,
    edge: &SuccessorData,
    facts: &mut PathFacts,
) -> Option<SuccessorData> {
    // A redirected edge cannot supply definitions used directly in other
    // blocks. Values passed explicitly as successor arguments can be remapped.
    let escapes = |f: &FuncBody, value| {
        f.dfg()
            .uses(value)
            .any(|site| f.layout().inst_block(site.inst()) != Some(edge.block))
    };
    for (&param, &arg) in f.dfg().block_params(edge.block).iter().zip(&edge.args) {
        if escapes(f, param) {
            return None;
        }
        facts.bind(param, arg);
    }
    let mut cursor = f.layout().first_inst(edge.block);
    while let Some(inst) = cursor {
        let view = f.dfg().inst(inst);
        if view.is_terminator() {
            return resolve_successor(f, edge.block, view, facts);
        }
        if f.dfg().inst_results(inst).iter().any(|&v| escapes(f, v)) {
            return None;
        }
        if !fold_or_skip(f, decls, inst, facts) {
            return None;
        }
        cursor = f.layout().next_inst(inst);
    }
    None
}

fn resolve_successor(
    f: &FuncBody,
    block: Block,
    view: InstView<'_>,
    facts: &PathFacts,
) -> Option<SuccessorData> {
    let dest = match view {
        InstView::Jump { dest } => dest,
        _ => {
            let Successors::One(index) = rewrite::successors(view, |v| facts.fact(f.dfg(), v))
            else {
                return None;
            };
            view.successor(index)?
        }
    };
    let mut target = SuccessorData::new(dest.block, &[]);
    for &arg in dest.args {
        let value = facts.resolve(arg);
        // Bypassing this block skips its definitions on the new edge.
        if f.dfg()
            .value_inst(value)
            .and_then(|i| f.layout().inst_block(i))
            == Some(block)
        {
            return None;
        }
        target.args.push(value);
    }
    Some(target)
}

/// Bind folded results, or check that the unfolded instruction can be skipped.
fn fold_or_skip(f: &mut FuncBody, decls: &FuncDecls, inst: Inst, facts: &mut PathFacts) -> bool {
    let view = f.dfg().inst(inst);
    let results = f.dfg().inst_results(inst);
    let types: SmallVec<[_; 2]> = results.iter().map(|&v| f.dfg().value_type(v)).collect();
    let args: SmallVec<[_; 4]> = f
        .dfg()
        .operands(inst)
        .iter()
        .map(|&v| facts.resolve(v))
        .collect();
    let folds = rewrite::fold(f.dfg().inst_fields(inst), &args, &types, |v| {
        facts.fact(f.dfg(), v).constant()
    })
    .or_else(|| {
        let InstView::IntCompare { kind, .. } = view else {
            return None;
        };
        let truth = facts.comparison(f.dfg(), kind, [args[0], args[1]])?;
        Some(smallvec![Fold::Constant(ScalarConst::from(truth))])
    });
    let Some(folds) = folds else {
        return view.can_erase(decls);
    };
    for (i, fold) in folds.into_iter().enumerate() {
        let result = f.dfg().inst_results(inst)[i];
        let value = match fold {
            Fold::Operand(index) => args[index],
            Fold::Constant(value) => f.edit().constant(value.into()),
        };
        facts.bind(result, value);
    }
    true
}
