//! Replace affine induction products with modular recurrences.
//! Width is unchanged: wrapping i32 induction never becomes an unbounded pointer
//! increment. Extensions and pointer formation remain after the recurrence.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use veloc_analyzer::{AnalysisManager, graph::DominatorTree};
use veloc_mir::{FuncBody, InstView, Opcode, Value, ValueDef, function::EdgeRef};

pub struct StrengthPass;
impl FunctionPass for StrengthPass {
    fn name(&self) -> &'static str {
        "StrengthPass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        _: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let f = am.function_mut();
        let dom = DominatorTree::compute(f.cfg(), f.entry_block());
        let insts: Vec<_> = f
            .layout()
            .block_order()
            .flat_map(|b| f.layout().block_insts(b))
            .collect();
        let mut changed = 0;
        for inst in insts {
            let InstView::Binary {
                opcode: op @ (Opcode::IMul | Opcode::IAdd),
                args: [a, b],
            } = f.dfg().inst(inst)
            else {
                continue;
            };
            let (a, b) = (*a, *b);
            for (induction, factor) in [(a, b), (b, a)] {
                let ValueDef::Param(header) = f.dfg().value_def(induction) else {
                    continue;
                };
                if header == f.entry_block() {
                    continue;
                }
                let position = f
                    .dfg()
                    .block_params(header)
                    .iter()
                    .position(|&p| p == induction)
                    .unwrap();
                let result = f.dfg().first_result(inst).unwrap();
                // Do not manufacture a second induction variable for the
                // induction update itself or a cheap constant offset.
                if op == Opcode::IAdd
                    && (f.dfg().as_scalar_const(factor).is_some()
                        || f.cfg().preds(header).iter().any(|&p| {
                            let mut update = false;
                            f.dfg()
                                .inst(f.layout().last_inst(p).unwrap())
                                .visit_successors(|e| {
                                    if e.block == header && e.args[position] == result {
                                        update = true;
                                    }
                                });
                            update
                        }))
                {
                    continue;
                }
                let preds = f.cfg().preds(header).to_vec();
                let entries: Vec<_> = preds
                    .iter()
                    .copied()
                    .filter(|&p| !dom.dominates(header, p))
                    .collect();
                let [entry] = entries[..] else {
                    continue;
                };
                if !available(f, factor, entry, &dom) {
                    continue;
                }
                let mut edges = Vec::new();
                let mut valid = true;
                let mut backs = 0;
                for pred in preds {
                    let last = f.layout().last_inst(pred).unwrap();
                    let mut index = 0;
                    f.dfg().inst(last).visit_successors(|edge| {
                        if edge.block == header {
                            let arg = edge.args[position];
                            let step = if dom.dominates(header, pred) {
                                backs += 1;
                                let step =
                                    f.dfg().value_inst(arg).and_then(|i| match f.dfg().inst(i) {
                                        InstView::Binary {
                                            opcode: Opcode::IAdd,
                                            args: [a, b],
                                        } if *a == induction => Some(*b),
                                        InstView::Binary {
                                            opcode: Opcode::IAdd,
                                            args: [a, b],
                                        } if *b == induction => Some(*a),
                                        _ => None,
                                    });
                                if step.is_none_or(|v| !available(f, v, entry, &dom)) {
                                    valid = false;
                                }
                                step
                            } else {
                                None
                            };
                            edges.push((
                                EdgeRef { inst: last, index },
                                arg,
                                step,
                                edge.args.to_vec(),
                            ));
                        }
                        index += 1;
                    });
                }
                if !valid || backs == 0 {
                    continue;
                }
                let ty = f.dfg().value_type(induction);
                let product = f.dfg().first_result(inst).unwrap();
                let recurrence = f.edit().append_block_param(header, ty);
                let anchor = f.layout().last_inst(entry).unwrap();
                for (edge, start, step, mut args) in edges {
                    let next = if let Some(step) = step {
                        let increment = if op == Opcode::IMul {
                            let i = f.edit().insert_before(
                                anchor,
                                |w| w.binary(Opcode::IMul, [step, factor]),
                                &[ty],
                            );
                            f.dfg().first_result(i).unwrap()
                        } else {
                            step
                        };
                        let next = f.edit().insert_before(
                            edge.inst,
                            |w| w.binary(Opcode::IAdd, [recurrence, increment]),
                            &[ty],
                        );
                        f.dfg().first_result(next).unwrap()
                    } else {
                        let initial = f.edit().insert_before(
                            edge.inst,
                            |w| w.binary(op, [start, factor]),
                            &[ty],
                        );
                        f.dfg().first_result(initial).unwrap()
                    };
                    args.push(next);
                    f.edit().redirect_edge(edge, header, &args);
                }
                f.edit().replace_all_uses(product, recurrence);
                f.edit().erase_inst(inst);
                changed += 1;
                break;
            }
        }
        metrics.count("strength.recurrences", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}
fn available(
    f: &FuncBody,
    v: Value,
    at: veloc_mir::Block,
    dom: &DominatorTree<veloc_mir::Block>,
) -> bool {
    match f.dfg().value_def(v) {
        ValueDef::Const(_) => true,
        ValueDef::Param(b) => dom.dominates(b, at),
        ValueDef::Inst(i) => f
            .layout()
            .inst_block(i)
            .is_some_and(|b| dom.dominates(b, at)),
    }
}
