//! Shared expression evaluation over operand identities and constant facts.
//! Spec supplies concrete semantics and local identities; callers own analysis
//! state, control-flow reachability and publication of the returned results.
use veloc_mir::constant::ScalarConst;
use veloc_mir::{InstFields, Opcode, Type, TypeInfo, Value};

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));

pub(crate) enum Fold {
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
}

/// A reusable operand identity or an abstract result fact. Operand indices
/// preserve the caller's original witnesses, including e-graph operands.
pub(crate) enum Evaluation {
    Operand(usize),
    Fact(Fact),
}

impl Evaluation {
    pub(crate) fn fact(self, operand: impl FnOnce(usize) -> Fact) -> Fact {
        match self {
            Self::Operand(i) => operand(i),
            Self::Fact(fact) => fact,
        }
    }
}

impl From<Fold> for Evaluation {
    fn from(fold: Fold) -> Self {
        match fold {
            Fold::Operand(i) => Self::Operand(i),
            Fold::Constant(c) => Self::Fact(Fact::Constant(c)),
        }
    }
}

/// Adapter for clients folding an existing instruction. The evaluator itself
/// also accepts an unallocated expression, so graph construction needs no MIR
/// node just to discover a constant.
pub fn fold(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: veloc_mir::Inst,
    mut constant: impl FnMut(Value) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[ScalarConst; 2]>> {
    let args: smallvec::SmallVec<[ScalarConst; 3]> = dfg
        .operands(inst)
        .iter()
        .map(|&value| {
            let c = constant(value)?;
            assert_eq!(dfg.value_type(value), c.ty(), "constant fact type");
            Some(c)
        })
        .collect::<Option<_>>()?;
    let results: smallvec::SmallVec<[Type; 2]> = dfg
        .inst_results(inst)
        .iter()
        .map(|&v| dfg.value_type(v))
        .collect();
    evaluate(dfg.inst_fields(inst), &args, &results)
}

mod local {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/local_folds.rs"));
}

pub(crate) use local::{accepts, plan, validate_fields};

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

/// Restrict folding to operations whose effects are modeled by the evaluator.
/// A potentially trapping operation may only be erased after a successful fold.
pub(crate) fn can_reduce(dfg: &veloc_mir::dfg::DataFlowGraph, inst: veloc_mir::Inst) -> bool {
    supports(
        dfg.opcode(inst),
        dfg.inst_results(inst).iter().map(|&v| dfg.value_type(v)),
    )
}

fn supports(opcode: Opcode, mut results: impl ExactSizeIterator<Item = Type>) -> bool {
    results.len() != 0
        && (local::can_fold(opcode)
            || (can_fold(opcode) && results.all(|ty| ScalarConst::from_bits(ty, 0).is_some())))
        && opcode.spec().memory_effect().is_none()
        && !opcode.spec().is_terminator()
        && !opcode.transfers_ownership()
}

/// Evaluate an existing instruction. Unsupported or effectful operations yield
/// Varying results even if some inputs are Unknown.
pub(crate) fn evaluate_inst(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: veloc_mir::Inst,
    fact: impl FnMut(Value) -> Fact,
) -> smallvec::SmallVec<[Evaluation; 2]> {
    let results: smallvec::SmallVec<[Type; 2]> = dfg
        .inst_results(inst)
        .iter()
        .map(|&v| dfg.value_type(v))
        .collect();
    evaluate_expression(dfg.inst_fields(inst), dfg.operands(inst), &results, fact)
}

/// Fold using only established constants, without abstract analysis facts.
pub(crate) fn reduce_inst(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: veloc_mir::Inst,
    constant: impl FnMut(Value) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    let results: smallvec::SmallVec<[Type; 2]> = dfg
        .inst_results(inst)
        .iter()
        .map(|&v| dfg.value_type(v))
        .collect();
    reduce(
        dfg.inst_fields(inst),
        dfg.operands(inst),
        &results,
        constant,
    )
}

/// Reduce an existing or proposed expression without constructing MIR nodes.
/// Operand indices refer to the original argument order; callers may supply
/// canonical values for equality checks while retaining their own SSA witnesses.
pub(crate) fn reduce<V: Copy + Eq>(
    fields: &InstFields,
    args: &[V],
    results: &[Type],
    constant: impl FnMut(V) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    if !supports(fields.opcode(), results.iter().copied()) {
        return None;
    }
    try_fold(fields, args, results, constant)
}

/// Identities use `V::eq`; abstract facts are queried separately. Returning an
/// operand keeps its identity even when its current fact is Unknown or Varying.
/// The caller supplies a well-typed expression, whether or not it exists in MIR.
pub(crate) fn evaluate_expression<V: Copy + Eq>(
    fields: &InstFields,
    args: &[V],
    results: &[Type],
    mut fact: impl FnMut(V) -> Fact,
) -> smallvec::SmallVec<[Evaluation; 2]> {
    let repeated = |fact| results.iter().map(|_| Evaluation::Fact(fact)).collect();
    if !supports(fields.opcode(), results.iter().copied()) {
        return repeated(Fact::Varying);
    }
    if let Some(folds) = try_fold(fields, args, results, |v| fact(v).constant()) {
        return folds.into_iter().map(Evaluation::from).collect();
    }
    let result = if fields.opcode() == Opcode::Select {
        // Spec already handled constant conditions and identical operands.
        // Join facts before the general Unknown check: a varying condition can
        // provisionally resolve a Select with one as-yet-unknown branch.
        match fact(args[0]) {
            Fact::Varying => fact(args[1]).join(fact(args[2])),
            Fact::Unknown => Fact::Unknown,
            Fact::Constant(_) => unreachable!("constant Select condition must fold"),
        }
    } else if args.iter().any(|&v| fact(v) == Fact::Unknown) {
        Fact::Unknown
    } else {
        Fact::Varying
    };
    repeated(result)
}

/// Concrete evaluation and identities shared by analysis and direct rewriting.
fn try_fold<V: Copy + Eq>(
    fields: &InstFields,
    args: &[V],
    results: &[Type],
    mut constant: impl FnMut(V) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    if let [ty] = results
        && let Some(fold) = local::fold(fields, *ty, args, &mut constant)
    {
        return Some(smallvec::smallvec![fold]);
    }
    let constants: smallvec::SmallVec<[ScalarConst; 3]> = args
        .iter()
        .map(|&value| constant(value))
        .collect::<Option<_>>()?;
    evaluate(fields, &constants, results)
        .map(|values| values.into_iter().map(Fold::Constant).collect())
}
