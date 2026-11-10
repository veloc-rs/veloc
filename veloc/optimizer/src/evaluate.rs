//! Shared expression evaluation over operand identities and constant facts.
//! Spec supplies concrete semantics and local identities; callers own analysis
//! state, control-flow reachability and publication of the returned results.
use veloc_mir::constant::ScalarConst;
use veloc_mir::{InstFields, Opcode, Type, TypeInfo, Value};

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));

/// A replacement for one instruction result. Operand indices refer to the
/// original argument order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Operand(usize),
    Constant(ScalarConst),
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

mod local {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/local_folds.rs"));
}

pub use local::can_fold;
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

/// Evaluate an existing instruction. Unsupported or effectful operations yield
/// Varying results even if some inputs are Unknown.
/// Identities compare SSA values; an operand fold yields that operand's current
/// fact, including Unknown or Varying.
pub(crate) fn evaluate_inst(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: veloc_mir::Inst,
    mut fact: impl FnMut(Value) -> Fact,
) -> smallvec::SmallVec<[Fact; 2]> {
    let results: smallvec::SmallVec<[Type; 2]> = dfg
        .inst_results(inst)
        .iter()
        .map(|&v| dfg.value_type(v))
        .collect();
    let fields = dfg.inst_fields(inst);
    let args = dfg.operands(inst);
    let repeated = |fact| results.iter().map(|_| fact).collect();
    if !can_fold(fields.opcode()) {
        return repeated(Fact::Varying);
    }
    if let Some(folds) = try_fold(fields, args, &results, |v| fact(v).constant()) {
        return folds
            .into_iter()
            .map(|fold| match fold {
                Fold::Operand(i) => fact(args[i]),
                Fold::Constant(c) => Fact::Constant(c),
            })
            .collect();
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

/// Fold an existing instruction to constants or original operands.
pub fn fold_inst(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: veloc_mir::Inst,
    constant: impl FnMut(Value) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    let results: smallvec::SmallVec<[Type; 2]> = dfg
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

/// Fold an existing or proposed expression without constructing MIR nodes.
/// Operand indices refer to the original argument order; callers may supply
/// canonical values for equality checks while retaining their own SSA witnesses.
/// The expression and supplied constants must be well typed. A potentially
/// trapping instruction may only be erased after a successful fold.
pub fn fold<V: Copy + Eq>(
    fields: &InstFields,
    args: &[V],
    results: &[Type],
    constant: impl FnMut(V) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    if !can_fold(fields.opcode()) {
        return None;
    }
    try_fold(fields, args, results, constant)
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
