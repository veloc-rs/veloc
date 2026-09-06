use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{BvOp, Error, IntPredicate, Outcome, Sort, Trap, Value, Width};

/// A typed immutable expression. Cloning shares nodes, including between functions.
#[derive(Clone, Debug)]
pub struct Expr(Arc<Term>);

#[derive(Debug)]
struct Term {
    sort: Sort,
    node: Node,
}

#[derive(Debug)]
pub(crate) enum Node {
    Input(usize),
    Bv(u128),
    Bool(bool),
    Apply {
        op: BvOp,
        args: Box<[Expr]>,
    },
    Concat(Expr, Expr),
    Extract {
        arg: Expr,
        high: u16,
        low: u16,
    },
    ZeroExtend(Expr),
    SignExtend(Expr),
    Compare {
        predicate: IntPredicate,
        lhs: Expr,
        rhs: Expr,
    },
    Select {
        cond: Expr,
        yes: Expr,
        no: Expr,
    },
}

impl Expr {
    fn new(sort: Sort, node: Node) -> Self {
        Self(Arc::new(Term { sort, node }))
    }

    /// Refer to a positional parameter. `Function::new` checks its declaration.
    pub fn input(index: usize, sort: Sort) -> Self {
        Self::new(sort, Node::Input(index))
    }

    /// Construct a bitvector literal, normalized to its declared width.
    pub fn bv(width: u16, value: u128) -> Result<Self, Error> {
        let width = Width::new(width)?;
        Ok(Self::new(Sort::Bv(width), Node::Bv(width.normalize(value))))
    }

    pub fn bool(value: bool) -> Self {
        Self::new(Sort::Bool, Node::Bool(value))
    }

    pub fn sort(&self) -> Sort {
        self.0.sort
    }

    pub fn apply(op: BvOp, args: &[Self]) -> Result<Self, Error> {
        if args.len() != op.arity() {
            return Err(Error::Arity {
                expected: op.arity(),
                actual: args.len(),
            });
        }
        let sort = args[0].sort();
        if !(sort == Sort::Bool && matches!(op, BvOp::And | BvOp::Or | BvOp::Xor)) {
            sort.width()?;
        }
        for arg in &args[1..] {
            check_sort(sort, arg.sort())?;
        }
        Ok(Self::new(
            sort,
            Node::Apply {
                op,
                args: args.into(),
            },
        ))
    }

    pub fn concat(high: &Self, low: &Self) -> Result<Self, Error> {
        let width = high.sort().width()?.bits() + low.sort().width()?.bits();
        Ok(Self::new(
            Sort::bv(width)?,
            Node::Concat(high.clone(), low.clone()),
        ))
    }

    /// Extract the inclusive range `[high:low]`, numbered from the least significant bit.
    pub fn extract(&self, high: u16, low: u16) -> Result<Self, Error> {
        let width = self.sort().width()?.bits();
        if low > high || high >= width {
            return Err(Error::InvalidExtract { width, high, low });
        }
        Ok(Self::new(
            Sort::bv(high - low + 1)?,
            Node::Extract {
                arg: self.clone(),
                high,
                low,
            },
        ))
    }

    pub fn zero_extend(&self, to: u16) -> Result<Self, Error> {
        let from = self.sort().width()?.bits();
        let sort = Sort::bv(to)?;
        if to < from {
            return Err(Error::InvalidExtension { from, to });
        }
        if to == from {
            return Ok(self.clone());
        }
        Ok(Self::new(sort, Node::ZeroExtend(self.clone())))
    }

    pub fn ult(&self, rhs: &Self) -> Result<Self, Error> {
        self.compare(rhs, IntPredicate::new(false, 1))
    }

    pub fn compare(&self, rhs: &Self, predicate: IntPredicate) -> Result<Self, Error> {
        self.sort().width()?;
        check_sort(self.sort(), rhs.sort())?;
        Ok(Self::new(
            Sort::Bool,
            Node::Compare {
                predicate,
                lhs: self.clone(),
                rhs: rhs.clone(),
            },
        ))
    }

    pub fn sign_extend(&self, to: u16) -> Result<Self, Error> {
        let from = self.sort().width()?.bits();
        let sort = Sort::bv(to)?;
        if to < from {
            return Err(Error::InvalidExtension { from, to });
        }
        if to == from {
            return Ok(self.clone());
        }
        Ok(Self::new(sort, Node::SignExtend(self.clone())))
    }

    pub fn select(cond: &Self, yes: &Self, no: &Self) -> Result<Self, Error> {
        check_sort(Sort::Bool, cond.sort())?;
        check_sort(yes.sort(), no.sort())?;
        Ok(Self::new(
            yes.sort(),
            Node::Select {
                cond: cond.clone(),
                yes: yes.clone(),
                no: no.clone(),
            },
        ))
    }

    /// Encode a boolean as a bitvector containing zero or one.
    pub fn bool_to_bv(&self, width: u16) -> Result<Self, Error> {
        Self::select(self, &Self::bv(width, 1)?, &Self::bv(width, 0)?)
    }

    pub(crate) fn key(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub(crate) fn node(&self) -> &Node {
        &self.0.node
    }

    fn visit_children(&self, mut visit: impl FnMut(&Self)) {
        match self.node() {
            Node::Apply { args, .. } => args.iter().for_each(visit),
            Node::Concat(a, b) | Node::Compare { lhs: a, rhs: b, .. } => {
                visit(a);
                visit(b);
            }
            Node::Extract { arg, .. } | Node::ZeroExtend(arg) | Node::SignExtend(arg) => visit(arg),
            Node::Select { cond, yes, no } => {
                visit(cond);
                visit(yes);
                visit(no);
            }
            Node::Input(_) | Node::Bv(_) | Node::Bool(_) => {}
        }
    }
}

fn check_sort(expected: Sort, actual: Sort) -> Result<(), Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::SortMismatch { expected, actual })
    }
}

/// A typed, multiple-result semantic function with a positional input signature.
///
/// Input signatures are explicit even for unused parameters. This makes execution
/// and equivalence checking agree about input positions, sorts, and widths.
#[derive(Clone, Debug)]
pub struct Function {
    inputs: Box<[Sort]>,
    pub(crate) nodes: Vec<Expr>,
    pub(crate) indices: HashMap<usize, usize>,
    outputs: Box<[Expr]>,
    traps: Box<[(Expr, Trap)]>,
}

impl Function {
    pub fn new(inputs: impl Into<Box<[Sort]>>, output: Expr) -> Result<Self, Error> {
        Self::with_outputs(inputs, [output])
    }

    pub fn with_outputs(
        inputs: impl Into<Box<[Sort]>>,
        outputs: impl Into<Box<[Expr]>>,
    ) -> Result<Self, Error> {
        Self::with_traps(inputs, outputs, [])
    }

    /// Guards are ordered: the first true guard selects the observable trap.
    /// All expressions are total; guarded-off result values are unobservable.
    pub fn with_traps(
        inputs: impl Into<Box<[Sort]>>,
        outputs: impl Into<Box<[Expr]>>,
        traps: impl Into<Box<[(Expr, Trap)]>>,
    ) -> Result<Self, Error> {
        let inputs = inputs.into();
        let outputs = outputs.into();
        let traps = traps.into();
        for (guard, _) in &traps {
            check_sort(Sort::Bool, guard.sort())?;
        }
        // Iterative postorder traversal preserves sharing without recursively
        // walking a potentially long expression chain.
        let mut stack = outputs
            .iter()
            .chain(traps.iter().map(|(guard, _)| guard))
            .rev()
            .cloned()
            .map(|expr| (expr, false))
            .collect::<Vec<_>>();
        let mut seen = HashSet::new();
        let mut nodes = Vec::new();
        let mut indices = HashMap::new();
        while let Some((expr, ready)) = stack.pop() {
            if ready {
                indices.insert(expr.key(), nodes.len());
                nodes.push(expr);
            } else if seen.insert(expr.key()) {
                if let Node::Input(index) = expr.node() {
                    let expected = *inputs.get(*index).ok_or(Error::InputIndex {
                        index: *index,
                        count: inputs.len(),
                    })?;
                    if expected != expr.sort() {
                        return Err(Error::InputSort {
                            index: *index,
                            expected,
                            actual: expr.sort(),
                        });
                    }
                }
                stack.push((expr.clone(), true));
                expr.visit_children(|child| stack.push((child.clone(), false)));
            }
        }
        Ok(Self {
            inputs,
            nodes,
            indices,
            outputs,
            traps,
        })
    }

    pub fn inputs(&self) -> &[Sort] {
        &self.inputs
    }

    pub fn output_sort(&self) -> Sort {
        assert_eq!(self.outputs.len(), 1, "expected a single-result function");
        self.outputs[0].sort()
    }

    pub fn outputs(&self) -> &[Expr] {
        &self.outputs
    }

    pub fn traps(&self) -> &[(Expr, Trap)] {
        &self.traps
    }

    pub fn eval(&self, args: &[Value]) -> Result<Value, Error> {
        if self.outputs.len() != 1 {
            return Err(Error::ResultArity {
                expected: 1,
                actual: self.outputs.len(),
            });
        }
        Ok(self.eval_all(args)?[0])
    }

    /// Evaluate with normalized bitvector inputs and exact boolean/bitvector sorts.
    pub fn eval_all(&self, args: &[Value]) -> Result<Vec<Value>, Error> {
        match self.execute(args)? {
            Outcome::Values(values) => Ok(values),
            Outcome::Trap(trap) => Err(Error::Trapped(trap)),
        }
    }

    /// Execute to an observable outcome. Err denotes an invalid call/graph,
    /// not a program trap. Value-only eval helpers report traps as Error::Trapped.
    pub fn execute(&self, args: &[Value]) -> Result<Outcome, Error> {
        if args.len() != self.inputs.len() {
            return Err(Error::Arity {
                expected: self.inputs.len(),
                actual: args.len(),
            });
        }
        let args = args
            .iter()
            .zip(self.inputs.iter())
            .enumerate()
            .map(|(index, (arg, sort))| match (arg, sort) {
                (Value::Bv(value), Sort::Bv(width)) => Ok(Value::Bv(width.normalize(*value))),
                (Value::Bool(value), Sort::Bool) => Ok(Value::Bool(*value)),
                _ => Err(Error::ValueSort {
                    index,
                    expected: *sort,
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut values = Vec::with_capacity(self.nodes.len());
        for expr in &self.nodes {
            let value = |expr: &Expr| values[self.indices[&expr.key()]];
            let bits = |expr: &Expr| match value(expr) {
                Value::Bv(bits) => bits,
                Value::Bool(_) => unreachable!("typed bitvector operand"),
            };
            let result = match expr.node() {
                Node::Input(index) => args[*index],
                Node::Bv(bits) => Value::Bv(*bits),
                Node::Bool(value) => Value::Bool(*value),
                Node::Apply { op, args } => {
                    if expr.sort() == Sort::Bool {
                        let args = args
                            .iter()
                            .map(|arg| match value(arg) {
                                Value::Bool(b) => u128::from(b),
                                _ => unreachable!("checked boolean operands"),
                            })
                            .collect::<Vec<_>>();
                        Value::Bool(op.eval(1, &args)? != 0)
                    } else {
                        let args = args.iter().map(bits).collect::<Vec<_>>();
                        Value::Bv(op.eval(expr.sort().width()?.bits(), &args)?)
                    }
                }
                Node::Concat(high, low) => {
                    // Both operands are nonzero width and their sum is at most
                    // 128, so this shift is always strictly less than 128.
                    Value::Bv((bits(high) << low.sort().width()?.bits()) | bits(low))
                }
                Node::Extract { arg, low, .. } => {
                    Value::Bv(expr.sort().width()?.normalize(bits(arg) >> low))
                }
                Node::ZeroExtend(arg) => Value::Bv(bits(arg)),
                Node::SignExtend(arg) => {
                    let from = arg.sort().width()?;
                    let value = bits(arg);
                    let extended = if value & (1 << (from.bits() - 1)) != 0 {
                        value | !from.mask()
                    } else {
                        value
                    };
                    Value::Bv(expr.sort().width()?.normalize(extended))
                }
                Node::Compare {
                    predicate,
                    lhs,
                    rhs,
                } => {
                    Value::Bool(predicate.eval(lhs.sort().width()?.bits(), bits(lhs), bits(rhs))?)
                }
                Node::Select { cond, yes, no } => match value(cond) {
                    Value::Bool(true) => value(yes),
                    Value::Bool(false) => value(no),
                    Value::Bv(_) => unreachable!("typed boolean condition"),
                },
            };
            values.push(result);
        }
        for (guard, trap) in &self.traps {
            if values[self.indices[&guard.key()]] == Value::Bool(true) {
                return Ok(Outcome::Trap(*trap));
            }
        }
        Ok(Outcome::Values(
            self.outputs
                .iter()
                .map(|expr| values[self.indices[&expr.key()]])
                .collect(),
        ))
    }
}
