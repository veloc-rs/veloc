//! Checked replacement plans for equality search. Construction folds direct
//! reductions before the graph allocates candidates or publishes equalities.
use crate::passes::expression::{ExprValue as Value, Expressions};
use smallvec::SmallVec;
use veloc_mir::{InstFields, ScalarConst, Type, TypeInfo};

/// Facts available to generated guards and replacement recipes.
pub(crate) struct Context<'a, 'ir> {
    pub body: &'a Expressions<'ir>,
    pub layout: Option<veloc_types::DataLayout>,
}
impl Context<'_, '_> {
    pub fn pointer_bits(&self) -> Option<u32> {
        self.layout.map(|l| u32::from(l.pointer_size) * 8)
    }
    pub fn ty(&self, value: Value) -> Type {
        self.body.value_type(value)
    }
    pub fn constant(&self, value: Value) -> Option<ScalarConst> {
        self.body.as_scalar_const(value)
    }
    pub fn fields(&self, value: Value) -> Option<&InstFields> {
        self.body.value_fields(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Input {
    Value(Value),
    Step(usize),
}

pub(crate) enum Step {
    Constant(ScalarConst),
    Build {
        ty: Type,
        args: SmallVec<[Input; 3]>,
        fields: InstFields,
    },
}

pub(crate) struct PlanBuilder {
    steps: SmallVec<[Step; 2]>,
}
impl Default for PlanBuilder {
    fn default() -> Self {
        Self {
            steps: SmallVec::new(),
        }
    }
}

pub(crate) struct Plan {
    steps: SmallVec<[Step; 2]>,
    result: Input,
}

impl PlanBuilder {
    pub fn ty(&self, cx: &Context<'_, '_>, input: Input) -> Type {
        match input {
            Input::Value(v) => cx.ty(v),
            Input::Step(i) => match self.steps[i] {
                Step::Constant(c) => c.ty(),
                Step::Build { ty, .. } => ty,
            },
        }
    }
    pub fn constant(&mut self, ty: Type, bits: u64) -> Option<Input> {
        let mask = u64::MAX.checked_shr(64u32.checked_sub(ty.element_bits()?)?)?;
        let c = ScalarConst::from_bits(ty, bits & mask)?;
        let result = Input::Step(self.steps.len());
        self.steps.push(Step::Constant(c));
        Some(result)
    }
    pub fn build(
        &mut self,
        cx: &Context<'_, '_>,
        fields: InstFields,
        ty: Type,
        args: &[Input],
    ) -> Option<Input> {
        let types: SmallVec<[Type; 3]> = args.iter().map(|&v| self.ty(cx, v)).collect();
        crate::evaluate::validate_fields(&fields, &types, &[ty])?;
        if let Some(folds) = crate::evaluate::reduce(&fields, args, &[ty], |input| match input {
            Input::Value(value) => cx.constant(value),
            Input::Step(i) => match self.steps[i] {
                Step::Constant(c) => Some(c),
                Step::Build { .. } => None,
            },
        }) {
            return match folds.into_iter().next().expect("single-result plan") {
                crate::evaluate::Fold::Operand(i) => Some(args[i]),
                crate::evaluate::Fold::Constant(c) => self.constant(c.ty(), c.to_bits()),
            };
        }
        if let Some(i) = self.steps.iter().position(|s| matches!(s, Step::Build { ty: t, args: a, fields: f } if *f == fields && *t == ty && a.as_slice() == args)) {
            return Some(Input::Step(i));
        }
        let result = Input::Step(self.steps.len());
        self.steps.push(Step::Build {
            ty,
            args: args.into(),
            fields,
        });
        Some(result)
    }
    pub fn finish(
        mut self,
        cx: &Context<'_, '_>,
        mut result: Input,
        expected: Type,
    ) -> Option<Plan> {
        if self.ty(cx, result) != expected {
            return None;
        }
        // Folding a parent can discard a whole planned subtree. Keep only the
        // reachable steps so discarded computations consume no graph budget.
        let mut live = SmallVec::<[bool; 4]>::new();
        live.resize(self.steps.len(), false);
        if let Input::Step(i) = result {
            live[i] = true;
        }
        for i in (0..self.steps.len()).rev() {
            if live[i]
                && let Step::Build { args, .. } = &self.steps[i]
            {
                for &arg in args {
                    if let Input::Step(j) = arg {
                        live[j] = true;
                    }
                }
            }
        }
        let mut remap = SmallVec::<[usize; 4]>::new();
        remap.resize(self.steps.len(), 0);
        let map = |input: &mut Input, remap: &[usize]| {
            if let Input::Step(i) = input {
                *i = remap[*i];
            }
        };
        let steps = core::mem::take(&mut self.steps);
        for (i, mut step) in steps.into_iter().enumerate() {
            if !live[i] {
                continue;
            }
            remap[i] = self.steps.len();
            if let Step::Build { args, .. } = &mut step {
                for arg in args {
                    map(arg, &remap);
                }
            }
            self.steps.push(step);
        }
        map(&mut result, &remap);
        Some(Plan {
            steps: self.steps,
            result,
        })
    }
}

impl Plan {
    pub fn materialize<E>(
        &self,
        mut emit: impl FnMut(&Step, &[Value]) -> Result<Value, E>,
    ) -> Result<Value, E> {
        fn value(input: Input, values: &[Value]) -> Value {
            match input {
                Input::Value(v) => v,
                Input::Step(i) => values[i],
            }
        }
        let mut values = SmallVec::<[Value; 2]>::new();
        for step in &self.steps {
            let args: SmallVec<[Value; 3]> = match step {
                Step::Build { args, .. } => args.iter().map(|&v| value(v, &values)).collect(),
                Step::Constant(_) => SmallVec::new(),
            };
            values.push(emit(step, &args)?);
        }
        Ok(value(self.result, &values))
    }
}
