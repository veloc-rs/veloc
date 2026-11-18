//! Scoped path constraints. Empty intersections reject a path; they are never
//! published as SCCP's Unknown. Undo records keep sibling paths independent.
use crate::{range::IntRange, rewrite::Fact};
use hashbrown::HashMap;
use veloc_mir::{FuncBody, InstView, IntCC, ScalarConst, Value, dfg::DataFlowGraph};

enum Undo {
    Value(Value, Option<Value>),
    Range(Value, Option<IntRange>),
    Comparison,
}

#[derive(Default)]
pub(crate) struct PathFacts {
    values: HashMap<Value, Value>,
    ranges: HashMap<Value, IntRange>,
    comparisons: Vec<(IntCC, [Value; 2])>,
    undo: Vec<Undo>,
}

impl PathFacts {
    pub(crate) fn checkpoint(&self) -> usize {
        self.undo.len()
    }

    pub(crate) fn resolve(&self, mut value: Value) -> Value {
        while let Some(&next) = self.values.get(&value) {
            value = next;
        }
        value
    }

    pub(crate) fn bind(&mut self, value: Value, replacement: Value) {
        let replacement = self.resolve(replacement);
        if value != replacement {
            let old = self.values.insert(value, replacement);
            self.undo.push(Undo::Value(value, old));
        }
    }

    pub(crate) fn fact(&self, dfg: &DataFlowGraph, value: Value) -> Fact {
        let value = self.resolve(value);
        if let Some(c) = dfg.as_scalar_const(value) {
            return Fact::Constant(c);
        }
        self.ranges.get(&value).map_or(Fact::Varying, |r| r.fact())
    }

    pub(crate) fn refine(&mut self, dfg: &DataFlowGraph, value: Value, allowed: IntRange) -> bool {
        let value = self.resolve(value);
        let range = self
            .fact(dfg, value)
            .range(dfg.value_type(value))
            .unwrap()
            .intersect(&allowed);
        if range.is_empty() {
            return false;
        }
        if self.ranges.get(&value) != Some(&range) {
            let old = self.ranges.insert(value, range);
            self.undo.push(Undo::Range(value, old));
        }
        true
    }

    pub(crate) fn comparison(
        &self,
        dfg: &DataFlowGraph,
        kind: IntCC,
        args: [Value; 2],
    ) -> Option<bool> {
        let args = args.map(|v| self.resolve(v));
        // Resolve saved operands at query time: later equalities may add aliases.
        for &(known, operands) in self.comparisons.iter().rev() {
            let operands = operands.map(|v| self.resolve(v));
            let query = if args == operands {
                kind
            } else if args == [operands[1], operands[0]] {
                kind.swap()
            } else {
                continue;
            };
            if let Some(answer) = known.implies(query) {
                return Some(answer);
            }
        }
        let [a, b] = args.map(|v| self.fact(dfg, v).range(dfg.value_type(v)));
        a?.compare(kind, &b?)
    }

    pub(crate) fn assume(&mut self, f: &mut FuncBody, condition: Value, truth: bool) -> bool {
        if let Some(c) = self.fact(f.dfg(), condition).constant()
            && (c.to_bits() != 0) != truth
        {
            return false;
        }
        let literal = f.edit().constant(ScalarConst::from(truth).into());
        self.bind(condition, literal);

        let dfg = f.dfg();
        let Some(InstView::IntCompare { kind, args }) = dfg.value_inst_view(condition) else {
            return true;
        };
        let kind = if truth { kind } else { kind.complement() };
        let args = args.map(|v| self.resolve(v));
        let [lhs, rhs] = args;
        if self.comparison(dfg, kind, args) == Some(false) {
            return false;
        }
        for (value, other, predicate) in [(lhs, rhs, kind), (rhs, lhs, kind.swap())] {
            if let Some(c) = self.fact(dfg, other).constant()
                && let Some(allowed) = IntRange::constraint(predicate, c)
                && !self.refine(dfg, value, allowed)
            {
                return false;
            }
        }
        if kind == IntCC::Eq && !self.bind_equal(dfg, lhs, rhs) {
            return false;
        }
        self.comparisons.push((kind, args));
        self.undo.push(Undo::Comparison);
        true
    }

    fn bind_equal(&mut self, dfg: &DataFlowGraph, lhs: Value, rhs: Value) -> bool {
        let [a, b] = [lhs, rhs].map(|v| self.fact(dfg, v).range(dfg.value_type(v)));
        let common = a.zip(b).map(|(a, b)| a.intersect(&b));
        if common.as_ref().is_some_and(IntRange::is_empty) {
            return false;
        }

        // Prefer a constant as the representative, otherwise use the smaller value ID.
        let (value, representative) = if dfg.as_scalar_const(lhs).is_some() {
            (rhs, lhs)
        } else if dfg.as_scalar_const(rhs).is_some() || rhs < lhs {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        self.bind(value, representative);
        if let Some(common) = common {
            self.refine(dfg, representative, common);
        }
        true
    }

    pub(crate) fn restore(&mut self, checkpoint: usize) {
        while self.undo.len() > checkpoint {
            match self.undo.pop().unwrap() {
                Undo::Value(value, old) => restore_entry(&mut self.values, value, old),
                Undo::Range(value, old) => restore_entry(&mut self.ranges, value, old),
                Undo::Comparison => {
                    self.comparisons.pop();
                }
            }
        }
    }
}

fn restore_entry<T>(map: &mut HashMap<Value, T>, value: Value, old: Option<T>) {
    if let Some(old) = old {
        map.insert(value, old);
    } else {
        map.remove(&value);
    }
}
