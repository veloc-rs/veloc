//! Dominator-scoped facts from conditional edges. A fact is installed only at
//! a unique-predecessor successor and undone on leaving its dominator subtree.
//! This exposes path-specific constants without cloning code or changing CFG.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use crate::{path_facts::PathFacts, rewrite::Fact};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, InstView, ScalarConst, Value};

pub struct PredicatePass;

fn incoming_fact(f: &FuncBody, block: Block) -> Option<(Value, bool)> {
    if block == f.entry_block() {
        return None;
    }
    let [pred] = f.cfg().preds(block) else {
        return None;
    };
    let InstView::Br {
        condition,
        then_dest,
        else_dest,
    } = f.dfg().inst(f.layout().last_inst(*pred)?)
    else {
        return None;
    };
    if then_dest.block == else_dest.block {
        return None;
    }
    Some((condition, then_dest.block == block))
}

impl FunctionPass for PredicatePass {
    fn name(&self) -> &'static str {
        "PredicatePass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let dom = am.take_dominators();
        let f = am.function_mut();
        let mut facts = PathFacts::default();
        // Explicit enter/leave events avoid recursion on deeply nested CFGs.
        let mut work = vec![(f.entry_block(), None)];
        let mut changed = 0;
        while let Some((block, checkpoint)) = work.pop() {
            if let Some(checkpoint) = checkpoint {
                facts.restore(checkpoint);
                continue;
            }
            let checkpoint = facts.checkpoint();
            if let Some((condition, truth)) = incoming_fact(f, block)
                && !facts.assume(f, condition, truth)
            {
                facts.restore(checkpoint);
                continue;
            }
            let insts: Vec<_> = f.layout().block_insts(block).collect();
            for inst in insts {
                let operands = f.dfg().operands(inst).to_vec();
                for (i, value) in operands.into_iter().enumerate() {
                    let mut new = facts.resolve(value);
                    if let Some(c) = facts.fact(f.dfg(), new).constant() {
                        new = f.edit().constant(c.into());
                    }
                    if value != new {
                        f.edit().set_operand(inst, i as u32, new);
                        changed += 1;
                    }
                }
                let fact = match f.dfg().inst(inst) {
                    InstView::IntCompare { kind, args } => facts
                        .comparison(f.dfg(), kind, *args)
                        .map(|truth| Fact::Constant(ScalarConst::from(truth))),
                    _ => None,
                }
                .or_else(|| crate::range::evaluate(f.dfg(), inst, |v| facts.fact(f.dfg(), v)));
                match fact {
                    Some(Fact::Constant(c)) => {
                        let result = f.dfg().first_result(inst).unwrap();
                        let literal = f.edit().constant(c.into());
                        facts.bind(result, literal);
                    }
                    Some(Fact::Range(range)) => {
                        let result = f.dfg().first_result(inst).unwrap();
                        facts.refine(f.dfg(), result, range);
                    }
                    _ => {}
                }
            }
            work.push((block, Some(checkpoint)));
            work.extend(dom.children(block).map(|child| (child, None)));
        }
        metrics.count("predicates.replaced_operands", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}
