//! Shared folding, expression plans and directed instruction rewrites.
//! Spec supplies semantics and rules; callers own analysis state and decide
//! when to apply the returned results.
use crate::passes::expression::{ExprValue, Expressions};
use smallvec::SmallVec;
use veloc_mir::{FuncBody, Inst, InstFields, Opcode, ScalarConst, Type, TypeInfo, Value};

/// A replacement for one instruction result. Operand indices refer to the
/// original argument order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Operand(usize),
    Constant(ScalarConst),
}

/// Constant-propagation lattice. Unknown means no information has arrived yet,
/// not an IR undef value. Facts never serve as operand identities.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Fact {
    #[default]
    Unknown,
    Constant(ScalarConst),
    Varying,
}

impl Fact {
    pub(crate) fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, x) | (x, Self::Unknown) => x,
            (a, b) if a == b => a,
            _ => Self::Varying,
        }
    }

    pub(crate) fn constant(self) -> Option<ScalarConst> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Select between facts after established operand/constant folds have run.
    /// A varying condition joins both arms; Unknown contributes no information
    /// yet, so a resulting constant remains provisional until the fixed point.
    pub(crate) fn select(self, yes: Self, no: Self) -> Self {
        match self {
            Self::Constant(c) => {
                if c.to_bits() != 0 {
                    yes
                } else {
                    no
                }
            }
            Self::Unknown => Self::Unknown,
            Self::Varying => yes.join(no),
        }
    }
}

/// Executable successor knowledge; Unknown selectors must not activate edges.
pub(crate) enum Successors {
    Pending,
    All,
    One(usize),
}

mod local {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/fold.rs"));
}

pub use local::{can_fold, fold};
pub(crate) use local::{successors, validate_fields};

/// Pattern literals are masked to the operand width. An unknown constant proves
/// neither equality nor inequality, in direct reductions, guards or queries.
pub(crate) fn matches_constant(value: Option<ScalarConst>, bits: u64, equal: bool) -> bool {
    value.is_some_and(|c| {
        c.ty()
            .element_bits()
            .and_then(|n| 64u32.checked_sub(n))
            .and_then(|shift| u64::MAX.checked_shr(shift))
            .is_some_and(|mask| (c.to_bits() == (bits & mask)) == equal)
    })
}

/// Repeated pattern variables accept identical values or identical established
/// constant facts. ScalarConst equality includes the type and exact bit pattern.
pub(crate) fn same_value<V: Copy + Eq>(
    lhs: V,
    rhs: V,
    mut constant: impl FnMut(V) -> Option<ScalarConst>,
) -> bool {
    lhs == rhs || constant(lhs).is_some_and(|value| constant(rhs) == Some(value))
}

/// Fold an existing instruction to constants or original operands.
pub fn fold_inst(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: Inst,
    constant: impl FnMut(Value) -> Option<ScalarConst>,
) -> Option<SmallVec<[Fold; 2]>> {
    let results: SmallVec<[Type; 2]> = dfg
        .inst_results(inst)
        .iter()
        .map(|&v| dfg.value_type(v))
        .collect();
    fold(
        dfg.inst_fields(inst),
        dfg.operands(inst),
        &results,
        constant,
    )
}

/// Facts available to generated guards and replacement recipes.
pub(crate) struct Context<'a, 'ir> {
    pub body: &'a Expressions<'ir>,
    pub layout: Option<veloc_types::DataLayout>,
}
impl Context<'_, '_> {
    pub fn pointer_bits(&self) -> Option<u32> {
        self.layout.map(|l| u32::from(l.pointer_size) * 8)
    }
    pub fn ty(&self, value: ExprValue) -> Type {
        self.body.value_type(value)
    }
    pub fn constant(&self, value: ExprValue) -> Option<ScalarConst> {
        self.body.as_scalar_const(value)
    }
    pub fn fields(&self, value: ExprValue) -> Option<&InstFields> {
        self.body.value_fields(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Input {
    Value(ExprValue),
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
        validate_fields(&fields, &types, &[ty])?;
        if let Some(folds) = fold(&fields, args, &[ty], |input| match input {
            Input::Value(value) => cx.constant(value),
            Input::Step(i) => match self.steps[i] {
                Step::Constant(c) => Some(c),
                Step::Build { .. } => None,
            },
        }) {
            return match folds.into_iter().next().expect("single-result plan") {
                Fold::Operand(i) => Some(args[i]),
                Fold::Constant(c) => self.constant(c.ty(), c.to_bits()),
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
        mut emit: impl FnMut(&Step, &[ExprValue]) -> Result<ExprValue, E>,
    ) -> Result<ExprValue, E> {
        fn value(input: Input, values: &[ExprValue]) -> ExprValue {
            match input {
                Input::Value(v) => v,
                Input::Step(i) => values[i],
            }
        }
        let mut values = SmallVec::<[ExprValue; 2]>::new();
        for step in &self.steps {
            let args: SmallVec<[ExprValue; 3]> = match step {
                Step::Build { args, .. } => args.iter().map(|&v| value(v, &values)).collect(),
                Step::Constant(_) => SmallVec::new(),
            };
            values.push(emit(step, &args)?);
        }
        Ok(value(self.result, &values))
    }
}

/// A checked edit for the matched instruction. Apply before changing its
/// operands or successors; matching itself does not modify the function.
pub(crate) enum InstRewrite {
    Replace {
        fields: InstFields,
        args: SmallVec<[Value; 3]>,
    },
    SelectEdge(u32),
    Branch {
        condition: Value,
        swap_edges: bool,
    },
}

impl InstRewrite {
    pub(crate) fn apply(self, func: &mut FuncBody, inst: Inst) {
        match self {
            Self::Replace { fields, args } => {
                func.edit()
                    .replace_inst(inst, |writer| writer.from_fields(fields, &args));
            }
            Self::SelectEdge(index) => {
                func.edit()
                    .fold_to_edge(veloc_mir::function::EdgeRef { inst, index });
            }
            Self::Branch {
                condition,
                swap_edges,
            } => {
                func.edit().rewrite_branch(inst, condition, swap_edges);
            }
        }
    }
}

mod directed {
    use super::InstRewrite;

    include!(concat!(env!("OUT_DIR"), "/rewrite.rs"));
}

pub(crate) use directed::find_rewrite;
