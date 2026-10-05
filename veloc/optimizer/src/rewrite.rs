//! Shared directed rewrites. Matching is immutable; each consumer decides how
//! to materialize the checked plan and publish the resulting equality.
use crate::evaluate::Properties;
use hashbrown::HashSet;
use smallvec::SmallVec;
use veloc_mir::{FuncBody, Inst, Opcode, ScalarConst, Type, TypeInfo, Value};

/// Read-only operation access shared by MIR and candidate storage.
pub(crate) trait View {
    type Value: Copy + Eq;
    fn ty(&self, value: Self::Value) -> Type;
    fn constant(&self, value: Self::Value) -> Option<ScalarConst>;
    fn properties(&self, value: Self::Value) -> Option<Properties>;
}

pub(crate) struct Node<'a> {
    pub opcode: Opcode,
    pub args: &'a [Value],
}

impl View for FuncBody {
    type Value = Value;
    fn ty(&self, value: Value) -> Type {
        self.dfg().value_type(value)
    }
    fn constant(&self, value: Value) -> Option<ScalarConst> {
        self.dfg().as_scalar_const(value)
    }
    fn properties(&self, value: Value) -> Option<Properties> {
        Some(Properties::read(
            self.dfg().inst(self.dfg().value_inst(value)?),
        ))
    }
}

impl Context<'_> {
    pub fn node(&self, value: Value) -> Option<Node<'_>> {
        let dfg = self.body.dfg();
        let inst = dfg.value_inst(value)?;
        let view = dfg.inst(inst);
        if dfg.inst_results(inst).len() != 1 || !view.can_speculate() {
            return None;
        }
        Some(Node {
            opcode: view.opcode(),
            args: dfg.operands(inst),
        })
    }
}

pub(crate) struct Context<'a, R: View + ?Sized = FuncBody> {
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
    pub fn matches_constant(&self, value: R::Value, bits: u64) -> bool {
        self.constant(value).is_some_and(|c| {
            c.ty()
                .element_bits()
                .and_then(|n| 64u32.checked_sub(n))
                .and_then(|shift| u64::MAX.checked_shr(shift))
                .is_some_and(|mask| c.to_bits() == (bits & mask))
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Input<V = Value> {
    Value(V),
    Step(usize),
}

pub(crate) enum Step<V = Value> {
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

pub(crate) struct Plan<V = Value> {
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
        // reachable steps so both hosts price and materialize the same work.
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

impl Plan {
    /// Price only computations that this replacement can actually remove.
    /// Shared definitions and values reused by the plan remain live.
    pub fn profitable(&self, body: &FuncBody, root: Value, canonical: bool) -> bool {
        if matches!(self.result, Input::Value(v) if v == root) {
            return false;
        }
        let Some(inst) = body.dfg().value_inst(root) else {
            return false;
        };
        let mut retained = HashSet::new();
        let mut retain = |input: Input| {
            if let Input::Value(v) = input {
                retained.insert(v);
            }
        };
        retain(self.result);
        let mut builds = 0;
        for step in &self.steps {
            if let Step::Build { args, .. } = step {
                builds += 1;
                for &arg in args {
                    retain(arg);
                }
            }
        }
        if retained.contains(&root) {
            return false;
        }
        if builds == 0 {
            return true;
        }
        let mut removed = HashSet::<Inst>::new();
        removed.insert(inst);
        let mut pending: Vec<_> = body.dfg().operands(inst).to_vec();
        while let Some(value) = pending.pop() {
            if retained.contains(&value) {
                continue;
            }
            let Some(def) = body.dfg().value_inst(value) else {
                continue;
            };
            if removed.contains(&def) || !body.dfg().inst(def).can_speculate() {
                continue;
            }
            if body.dfg().inst_results(def).iter().all(|v| {
                !retained.contains(v)
                    && body
                        .dfg()
                        .uses(*v)
                        .all(|site| removed.contains(&site.inst()))
            }) {
                removed.insert(def);
                pending.extend(body.dfg().operands(def));
            }
        }
        builds < removed.len() || (canonical && builds == removed.len())
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

/// Materialize before the matched instruction; the caller replaces its uses,
/// erases it and wakes all affected instructions in its worklist.
pub(crate) fn apply(f: &mut FuncBody, anchor: Inst, plan: &Plan) -> (Value, SmallVec<[Inst; 2]>) {
    let mut created = SmallVec::new();
    let value = plan
        .materialize::<core::convert::Infallible>(|step, args| {
            let value = match *step {
                Step::Constant(c) => f.edit().constant(c.into()),
                Step::Build {
                    opcode,
                    ty,
                    properties,
                    ..
                } => {
                    let inst = f.edit().insert_before(
                        anchor,
                        |w| properties.write(opcode, args, w),
                        &[ty],
                    );
                    created.push(inst);
                    f.dfg().first_result(inst).unwrap()
                }
            };
            Ok(value)
        })
        .unwrap();
    (value, created)
}
