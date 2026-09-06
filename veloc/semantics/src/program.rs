use crate::{BvConst, BvOp, Error, Expr, Function, Sort, Width};

/// A width-parameterized, single-result program over pure bitvectors.
///
/// The representation can be emitted as static data by a definition compiler.
/// Every input, constant, intermediate value and result has the selected width.
/// This does not model effects, traps, floating point, or varying lane widths.
/// Public fields are checked by [`Self::validate`] before execution or export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    pub inputs: u8,
    pub steps: &'static [Step],
    pub output: u16,
}

/// Steps are in dependency order. Apply operands index earlier steps, not inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Input(u8),
    Const(BvConst),
    Apply { op: BvOp, args: &'static [u16] },
}

impl Program {
    pub const fn arity(&self) -> usize {
        self.inputs as usize
    }

    /// Check all steps, including unreachable ones, and the output reference.
    pub fn validate(&self) -> Result<(), Error> {
        const MAX_STEPS: usize = u16::MAX as usize + 1;
        if self.steps.len() > MAX_STEPS {
            return Err(Error::TooManySteps {
                count: self.steps.len(),
                max: MAX_STEPS,
            });
        }
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                Step::Input(input) if *input >= self.inputs => {
                    return Err(Error::InputIndex {
                        index: usize::from(*input),
                        count: usize::from(self.inputs),
                    });
                }
                Step::Apply { op, args } => {
                    if args.len() != op.arity() {
                        return Err(Error::Arity {
                            expected: op.arity(),
                            actual: args.len(),
                        });
                    }
                    for &arg in *args {
                        if usize::from(arg) >= index {
                            return Err(Error::StepIndex {
                                index: usize::from(arg),
                                count: index,
                            });
                        }
                    }
                }
                Step::Input(_) | Step::Const(_) => {}
            }
        }
        if usize::from(self.output) >= self.steps.len() {
            return Err(Error::StepIndex {
                index: usize::from(self.output),
                count: self.steps.len(),
            });
        }
        Ok(())
    }

    /// Execute modulo `2^width`, normalizing inputs to the chosen width.
    pub fn eval(&self, width: Width, args: &[u128]) -> Result<u128, Error> {
        self.validate()?;
        if args.len() != usize::from(self.inputs) {
            return Err(Error::Arity {
                expected: usize::from(self.inputs),
                actual: args.len(),
            });
        }
        if let Some(op) = self.primitive_validated() {
            return op.eval(width.bits(), args);
        }
        let mut values = Vec::with_capacity(self.steps.len());
        for step in self.steps {
            let value = match step {
                Step::Input(input) => width.normalize(args[usize::from(*input)]),
                Step::Const(constant) => constant.eval(width.bits())?,
                Step::Apply { op, args } => {
                    // The primitive set is unary/binary. Keep operands on the
                    // stack instead of allocating for every application.
                    let mut operands = [0; BvOp::MAX_ARITY];
                    for (operand, &arg) in operands.iter_mut().zip(*args) {
                        *operand = values[usize::from(arg)];
                    }
                    op.eval(width.bits(), &operands[..args.len()])?
                }
            };
            values.push(value);
        }
        Ok(values[usize::from(self.output)])
    }

    /// Bind the width and translate into the graph used by execution and SMT.
    /// The complete input signature is preserved, including unused inputs.
    pub fn instantiate(&self, width: Width) -> Result<Function, Error> {
        self.validate()?;
        let sort = Sort::Bv(width);
        let mut expressions = Vec::<Expr>::with_capacity(self.steps.len());
        for step in self.steps {
            let expr = match step {
                Step::Input(input) => Expr::input(usize::from(*input), sort),
                Step::Const(constant) => Expr::bv(width.bits(), constant.eval(width.bits())?)?,
                Step::Apply { op, args } => {
                    let args = args
                        .iter()
                        .map(|&arg| expressions[usize::from(arg)].clone())
                        .collect::<Vec<_>>();
                    Expr::apply(*op, &args)?
                }
            };
            expressions.push(expr);
        }
        Function::new(
            vec![sort; usize::from(self.inputs)],
            expressions[usize::from(self.output)].clone(),
        )
    }

    /// Recognize only a direct primitive over every logical input, in order.
    ///
    /// This is structural recognition, not equivalence inference. For example,
    /// `sub(Zero, input(0))` is not classified as Neg. Valid unrelated steps are
    /// permitted; invalid steps make recognition fail even if unreachable.
    pub fn primitive(&self) -> Option<BvOp> {
        self.validate().ok()?;
        self.primitive_validated()
    }

    fn primitive_validated(&self) -> Option<BvOp> {
        let Step::Apply { op, args } = self.steps[usize::from(self.output)] else {
            return None;
        };
        if usize::from(self.inputs) != op.arity() {
            return None;
        }
        args.iter()
            .enumerate()
            .all(|(input, &arg)| self.steps[usize::from(arg)] == Step::Input(input as u8))
            .then_some(op)
    }
}
