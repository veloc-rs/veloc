use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{BvOp, Error, Width};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sort {
    Bool,
    Bv(Width),
}

impl Sort {
    pub fn bv(width: u16) -> Result<Self, Error> {
        Width::new(width).map(Self::Bv)
    }

    pub fn width(self) -> Result<Width, Error> {
        match self {
            Self::Bv(width) => Ok(width),
            Self::Bool => Err(Error::ExpectedBv(self)),
        }
    }
}

impl std::fmt::Display for Sort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool => f.write_str("bool"),
            Self::Bv(width) => write!(f, "bv{}", width.bits()),
        }
    }
}

/// Runtime values use the width declared in the function signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Bv(u128),
    Bool(bool),
}

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
    Apply { op: BvOp, args: Box<[Expr]> },
    Concat(Expr, Expr),
    Extract { arg: Expr, high: u16, low: u16 },
    ZeroExtend(Expr),
    Ult(Expr, Expr),
    Select { cond: Expr, yes: Expr, no: Expr },
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
        sort.width()?;
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
        self.sort().width()?;
        check_sort(self.sort(), rhs.sort())?;
        Ok(Self::new(Sort::Bool, Node::Ult(self.clone(), rhs.clone())))
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
            Node::Concat(a, b) | Node::Ult(a, b) => {
                visit(a);
                visit(b);
            }
            Node::Extract { arg, .. } | Node::ZeroExtend(arg) => visit(arg),
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

/// A single-result semantic function with a fixed, positional input signature.
///
/// Input signatures are explicit even for unused parameters. This makes execution
/// and equivalence checking agree about input positions, sorts, and widths.
#[derive(Clone, Debug)]
pub struct Function {
    inputs: Box<[Sort]>,
    pub(crate) nodes: Vec<Expr>,
    pub(crate) indices: HashMap<usize, usize>,
}

impl Function {
    pub fn new(inputs: impl Into<Box<[Sort]>>, output: Expr) -> Result<Self, Error> {
        let inputs = inputs.into();
        // Iterative postorder traversal preserves sharing without recursively
        // walking a potentially long expression chain.
        let mut stack = vec![(output, false)];
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
        })
    }

    pub fn inputs(&self) -> &[Sort] {
        &self.inputs
    }

    pub fn output_sort(&self) -> Sort {
        self.nodes.last().expect("a function has an output").sort()
    }

    /// Evaluate with normalized bitvector inputs and exact boolean/bitvector sorts.
    pub fn eval(&self, args: &[Value]) -> Result<Value, Error> {
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
                    let args = args.iter().map(bits).collect::<Vec<_>>();
                    Value::Bv(op.eval(expr.sort().width()?.bits(), &args)?)
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
                Node::Ult(a, b) => Value::Bool(bits(a) < bits(b)),
                Node::Select { cond, yes, no } => match value(cond) {
                    Value::Bool(true) => value(yes),
                    Value::Bool(false) => value(no),
                    Value::Bv(_) => unreachable!("typed boolean condition"),
                },
            };
            values.push(result);
        }
        Ok(*values.last().expect("a function has an output"))
    }
}
