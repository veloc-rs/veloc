use crate::{BvConst, BvOp, Error, Expr, Function, IntPredicate, Sort, Trap};

/// A sort supplied by the operation signature, independent of storage layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeRef {
    Input(u8),
    Result(u8),
    Fixed(Sort),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonRef {
    Property(u8),
    Fixed(IntPredicate),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    ZeroExtend,
    SignExtend,
    Truncate,
}

/// A static/borrowed recipe for a typed semantic graph. Signature sorts and
/// compile-time comparison properties are bound before evaluation or SMT export.
/// Vector operations may reuse a recipe per lane; this is not a vector model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Program<'a, A = &'a [u16]> {
    pub inputs: u8,
    pub properties: u8,
    pub steps: &'a [Step<A>],
    pub outputs: &'a [u16],
    pub traps: &'a [(u16, Trap)],
}

/// Offline semantic description keyed by a consumer's operation identifier.
/// The consumer owns opcode/type/effect metadata; this crate only owns the recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticSpec<O> {
    pub opcode: O,
    pub program: Program<'static>,
}

/// An encoding adapter, not a second evaluator: every step is constructed
/// through the same checked Expr API used by dynamically constructed graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step<A = &'static [u16]> {
    Input(u8),
    Const {
        value: BvConst,
        ty: TypeRef,
    },
    Apply {
        op: BvOp,
        args: A,
    },
    Compare {
        kind: ComparisonRef,
        lhs: u16,
        rhs: u16,
    },
    Convert {
        kind: Conversion,
        arg: u16,
        to: TypeRef,
    },
    Select {
        cond: u16,
        yes: u16,
        no: u16,
    },
}

impl<A: AsRef<[u16]>> Program<'_, A> {
    pub const fn arity(&self) -> usize {
        self.inputs as usize
    }

    pub fn validate(&self) -> Result<(), Error> {
        const MAX: usize = u16::MAX as usize + 1;
        if self.steps.len() > MAX {
            return Err(Error::TooManySteps {
                count: self.steps.len(),
                max: MAX,
            });
        }
        let ty = |ty: TypeRef| match ty {
            TypeRef::Input(i) if i >= self.inputs => Err(Error::InputIndex {
                index: i as usize,
                count: self.inputs as usize,
            }),
            TypeRef::Result(i) if i as usize >= self.outputs.len() => Err(Error::ResultArity {
                expected: i as usize + 1,
                actual: self.outputs.len(),
            }),
            _ => Ok(()),
        };
        for (index, step) in self.steps.iter().enumerate() {
            let reference = |arg: u16| {
                if arg as usize >= index {
                    Err(Error::StepIndex {
                        index: arg as usize,
                        count: index,
                    })
                } else {
                    Ok(())
                }
            };
            match step {
                Step::Input(i) => ty(TypeRef::Input(*i))?,
                Step::Const { ty: sort, .. } => ty(*sort)?,
                Step::Apply { op, args } => {
                    if args.as_ref().len() != op.arity() {
                        return Err(Error::Arity {
                            expected: op.arity(),
                            actual: args.as_ref().len(),
                        });
                    }
                    for &arg in args.as_ref() {
                        reference(arg)?;
                    }
                }
                Step::Compare { kind, lhs, rhs } => {
                    reference(*lhs)?;
                    reference(*rhs)?;
                    if let ComparisonRef::Property(i) = kind
                        && *i >= self.properties
                    {
                        return Err(Error::PropertyIndex {
                            index: *i as usize,
                            count: self.properties as usize,
                        });
                    }
                }
                Step::Convert { arg, to, .. } => {
                    reference(*arg)?;
                    ty(*to)?;
                }
                Step::Select { cond, yes, no } => {
                    reference(*cond)?;
                    reference(*yes)?;
                    reference(*no)?;
                }
            }
        }
        for output in self
            .outputs
            .iter()
            .copied()
            .chain(self.traps.iter().map(|&(guard, _)| guard))
        {
            if output as usize >= self.steps.len() {
                return Err(Error::StepIndex {
                    index: output as usize,
                    count: self.steps.len(),
                });
            }
        }
        Ok(())
    }

    pub fn instantiate(
        &self,
        inputs: &[Sort],
        results: &[Sort],
        properties: &[IntPredicate],
    ) -> Result<Function, Error> {
        self.validate()?;
        if inputs.len() != self.inputs as usize {
            return Err(Error::Arity {
                expected: self.inputs as usize,
                actual: inputs.len(),
            });
        }
        if results.len() != self.outputs.len() {
            return Err(Error::ResultArity {
                expected: self.outputs.len(),
                actual: results.len(),
            });
        }
        if properties.len() != self.properties as usize {
            return Err(Error::Arity {
                expected: self.properties as usize,
                actual: properties.len(),
            });
        }
        let sort = |ty: TypeRef| match ty {
            TypeRef::Input(i) => inputs[i as usize],
            TypeRef::Result(i) => results[i as usize],
            TypeRef::Fixed(sort) => sort,
        };
        let mut expressions = Vec::<Expr>::with_capacity(self.steps.len());
        for step in self.steps {
            let get = |i: u16| &expressions[i as usize];
            let expr = match step {
                Step::Input(i) => Expr::input(*i as usize, inputs[*i as usize]),
                Step::Const { value, ty } => match sort(*ty) {
                    Sort::Bool => Expr::bool(value.eval(1)? != 0),
                    Sort::Bv(width) => Expr::bv(width.bits(), value.eval(width.bits())?)?,
                },
                Step::Apply { op, args } => Expr::apply(
                    *op,
                    &args
                        .as_ref()
                        .iter()
                        .map(|&i| get(i).clone())
                        .collect::<Vec<_>>(),
                )?,
                Step::Compare { kind, lhs, rhs } => get(*lhs).compare(
                    get(*rhs),
                    match kind {
                        ComparisonRef::Property(i) => properties[*i as usize],
                        ComparisonRef::Fixed(p) => *p,
                    },
                )?,
                Step::Convert { kind, arg, to } => {
                    let arg = get(*arg);
                    let width = sort(*to).width()?.bits();
                    match kind {
                        Conversion::ZeroExtend if arg.sort() == Sort::Bool => {
                            arg.bool_to_bv(width)?
                        }
                        Conversion::ZeroExtend => arg.zero_extend(width)?,
                        Conversion::SignExtend => arg.sign_extend(width)?,
                        Conversion::Truncate => arg.extract(width - 1, 0)?,
                    }
                }
                Step::Select { cond, yes, no } => Expr::select(get(*cond), get(*yes), get(*no))?,
            };
            expressions.push(expr);
        }
        let outputs = self
            .outputs
            .iter()
            .zip(results)
            .map(|(&i, &expected)| {
                let expr = expressions[i as usize].clone();
                if expr.sort() != expected {
                    Err(Error::SortMismatch {
                        expected,
                        actual: expr.sort(),
                    })
                } else {
                    Ok(expr)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let traps = self
            .traps
            .iter()
            .map(|&(guard, trap)| (expressions[guard as usize].clone(), trap))
            .collect::<Vec<_>>();
        Function::with_traps(inputs.to_vec(), outputs, traps)
    }

    /// Structural recognition only; never a proof of algebraic equivalence.
    pub fn primitive(&self) -> Option<BvOp> {
        self.validate().ok()?;
        if self.properties != 0 || !self.traps.is_empty() {
            return None;
        }
        let [output] = self.outputs else {
            return None;
        };
        let Step::Apply { op, args } = &self.steps[*output as usize] else {
            return None;
        };
        if self.inputs as usize != op.arity() {
            return None;
        }
        args.as_ref().iter().enumerate().all(|(i, &arg)|
            matches!(self.steps[arg as usize], Step::Input(input) if input as usize == i)
        ).then_some(*op)
    }
}
