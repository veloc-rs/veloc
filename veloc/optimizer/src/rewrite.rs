//! Checked replacement plans for equality search. Construction folds direct
//! reductions before the graph allocates candidates or publishes equalities.
use crate::evaluate::Properties;
use smallvec::SmallVec;
use veloc_mir::{Opcode, ScalarConst, Type, TypeInfo};

/// Facts available to generated guards and replacement recipes.
pub(crate) trait View {
    type Value: Copy + Eq;
    fn ty(&self, value: Self::Value) -> Type;
    fn constant(&self, value: Self::Value) -> Option<ScalarConst>;
    fn properties(&self, value: Self::Value) -> Option<Properties>;
}

pub(crate) struct Context<'a, R: View + ?Sized> {
    pub body: &'a R,
    pub layout: Option<veloc_types::DataLayout>,
}
impl<R: View + ?Sized> Context<'_, R> {
    pub fn pointer_bits(&self) -> Option<u32> {
        self.layout.map(|l| u32::from(l.pointer_size) * 8)
    }
    pub fn ty(&self, value: R::Value) -> Type {
        self.body.ty(value)
    }
    pub fn constant(&self, value: R::Value) -> Option<ScalarConst> {
        self.body.constant(value)
    }
    pub fn properties(&self, value: R::Value) -> Option<Properties> {
        self.body.properties(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Input<V> {
    Value(V),
    Step(usize),
}

pub(crate) enum Step<V> {
    Constant(ScalarConst),
    Build {
        opcode: Opcode,
        ty: Type,
        args: SmallVec<[Input<V>; 3]>,
        properties: Properties,
    },
}

pub(crate) struct PlanBuilder<V> {
    steps: SmallVec<[Step<V>; 2]>,
}
impl<V> Default for PlanBuilder<V> {
    fn default() -> Self {
        Self {
            steps: SmallVec::new(),
        }
    }
}

pub(crate) struct Plan<V> {
    steps: SmallVec<[Step<V>; 2]>,
    result: Input<V>,
}

impl<V: Copy + Eq> PlanBuilder<V> {
    pub fn ty(&self, cx: &Context<'_, impl View<Value = V> + ?Sized>, input: Input<V>) -> Type {
        match input {
            Input::Value(v) => cx.ty(v),
            Input::Step(i) => match self.steps[i] {
                Step::Constant(c) => c.ty(),
                Step::Build { ty, .. } => ty,
            },
        }
    }
    pub fn constant(&mut self, ty: Type, bits: u64) -> Option<Input<V>> {
        let mask = u64::MAX.checked_shr(64u32.checked_sub(ty.element_bits()?)?)?;
        let c = ScalarConst::from_bits(ty, bits & mask)?;
        let result = Input::Step(self.steps.len());
        self.steps.push(Step::Constant(c));
        Some(result)
    }
    pub fn build(
        &mut self,
        cx: &Context<'_, impl View<Value = V> + ?Sized>,
        opcode: Opcode,
        ty: Type,
        args: &[Input<V>],
        properties: Properties,
    ) -> Option<Input<V>> {
        let types: SmallVec<[Type; 3]> = args.iter().map(|&v| self.ty(cx, v)).collect();
        properties.validate(opcode, &types, &[ty])?;
        if let Some(folds) =
            crate::evaluate::reduce(opcode, args, &[ty], &properties, |input| match input {
                Input::Value(value) => cx.constant(value),
                Input::Step(i) => match self.steps[i] {
                    Step::Constant(c) => Some(c),
                    Step::Build { .. } => None,
                },
            })
        {
            return match folds.into_iter().next().expect("single-result plan") {
                crate::evaluate::Fold::Operand(i) => Some(args[i]),
                crate::evaluate::Fold::Constant(c) => self.constant(c.ty(), c.to_bits()),
            };
        }
        if let Some(i) = self.steps.iter().position(|s| matches!(s, Step::Build { opcode: op, ty: t, args: a, properties: p } if *op == opcode && *p == properties && *t == ty && a.as_slice() == args)) {
            return Some(Input::Step(i));
        }
        let result = Input::Step(self.steps.len());
        self.steps.push(Step::Build {
            opcode,
            ty,
            args: args.into(),
            properties,
        });
        Some(result)
    }
    pub fn finish(
        mut self,
        cx: &Context<'_, impl View<Value = V> + ?Sized>,
        mut result: Input<V>,
        expected: Type,
    ) -> Option<Plan<V>> {
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
        let map = |input: &mut Input<V>, remap: &[usize]| {
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

impl<V: Copy> Plan<V> {
    pub fn materialize<E>(
        &self,
        mut emit: impl FnMut(&Step<V>, &[V]) -> Result<V, E>,
    ) -> Result<V, E> {
        fn value<V: Copy>(input: Input<V>, values: &[V]) -> V {
            match input {
                Input::Value(v) => v,
                Input::Step(i) => values[i],
            }
        }
        let mut values = SmallVec::<[V; 2]>::new();
        for step in &self.steps {
            let args: SmallVec<[V; 3]> = match step {
                Step::Build { args, .. } => args.iter().map(|&v| value(v, &values)).collect(),
                Step::Constant(_) => SmallVec::new(),
            };
            values.push(emit(step, &args)?);
        }
        Ok(value(self.result, &values))
    }
}
