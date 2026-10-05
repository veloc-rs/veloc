//! Dominator-scoped facts from conditional edges. A fact is installed only at
//! a unique-predecessor successor and undone on leaving its dominator subtree.
//! This exposes path-specific constants without cloning code or changing CFG.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, InstView, IntCC, ScalarConst, Type, Value};

pub struct PredicatePass;

enum Undo {
    Constant(Value, Option<Value>),
    Comparison([Value; 2]),
}
#[derive(Default)]
struct Facts {
    constants: HashMap<Value, Value>,
    comparisons: HashMap<[Value; 2], Vec<IntCC>>,
    undo: Vec<Undo>,
}
impl Facts {
    fn constant(&mut self, value: Value, literal: Value) {
        if value != literal {
            let old = self.constants.insert(value, literal);
            self.undo.push(Undo::Constant(value, old));
        }
    }
    fn resolve(&self, value: Value) -> Value {
        self.constants.get(&value).copied().unwrap_or(value)
    }
    fn assume(&mut self, f: &mut FuncBody, condition: Value, truth: bool) {
        let literal = f.edit().constant(
            ScalarConst::from_bits(Type::BOOL, u64::from(truth))
                .unwrap()
                .into(),
        );
        self.constant(condition, literal);
        let Some(inst) = f.dfg().value_inst(condition) else {
            return;
        };
        if let InstView::IntCompare { mut kind, args } = f.dfg().inst(inst) {
            let mut args = args.map(|v| self.resolve(v));
            if !truth {
                kind = kind.complement();
            }
            if kind == IntCC::Eq {
                if f.dfg().as_scalar_const(args[0]).is_some() {
                    self.constant(args[1], args[0]);
                } else if f.dfg().as_scalar_const(args[1]).is_some() {
                    self.constant(args[0], args[1]);
                }
            }
            if args[1] < args[0] {
                args.swap(0, 1);
                kind = kind.swap();
            }
            self.undo.push(Undo::Comparison(args));
            self.comparisons.entry(args).or_default().push(kind);
        }
    }
    fn comparison(&self, mut args: [Value; 2], mut kind: IntCC) -> Option<bool> {
        if args[1] < args[0] {
            args.swap(0, 1);
            kind = kind.swap();
        }
        self.comparisons
            .get(&args)?
            .iter()
            .find_map(|known| known.implies(kind))
    }
    fn restore(&mut self, checkpoint: usize) {
        while self.undo.len() > checkpoint {
            match self.undo.pop().unwrap() {
                Undo::Constant(value, old) => {
                    if let Some(old) = old {
                        self.constants.insert(value, old);
                    } else {
                        self.constants.remove(&value);
                    }
                }
                Undo::Comparison(args) => {
                    let predicates = self.comparisons.get_mut(&args).unwrap();
                    predicates.pop();
                    if predicates.is_empty() {
                        self.comparisons.remove(&args);
                    }
                }
            }
        }
    }
}

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
        let mut facts = Facts::default();
        // Explicit enter/leave events avoid recursion on deeply nested CFGs.
        let mut work = vec![(f.entry_block(), None)];
        let mut changed = 0;
        while let Some((block, checkpoint)) = work.pop() {
            if let Some(checkpoint) = checkpoint {
                facts.restore(checkpoint);
                continue;
            }
            let checkpoint = facts.undo.len();
            if let Some((condition, truth)) = incoming_fact(f, block) {
                facts.assume(f, condition, truth);
            }
            let insts: Vec<_> = f.layout().block_insts(block).collect();
            for inst in insts {
                let operands = f.dfg().operands(inst).to_vec();
                for (i, value) in operands.into_iter().enumerate() {
                    let new = facts.resolve(value);
                    if value != new {
                        f.edit().set_operand(inst, i as u32, new);
                        changed += 1;
                    }
                }
                if let InstView::IntCompare { kind, args } = f.dfg().inst(inst)
                    && let Some(truth) = facts.comparison(*args, kind)
                {
                    let result = f.dfg().first_result(inst).unwrap();
                    let literal = f.edit().constant(
                        ScalarConst::from_bits(Type::BOOL, u64::from(truth))
                            .unwrap()
                            .into(),
                    );
                    facts.constant(result, literal);
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
