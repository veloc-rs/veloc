//! Shared constant evaluation and root-local folds generated from Spec.
use veloc_mir::constant::ScalarConst;
use veloc_mir::{IntCC, Opcode, Type, Value};

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));

pub(crate) enum Fold {
    Operand(usize),
    Constant(ScalarConst),
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
    let data = dfg.inst(inst);
    evaluate(data.opcode(), &args, &results, &properties(&data))
}

mod local {
    use super::*;
    use veloc_types::TypeInfo;
    include!(concat!(env!("OUT_DIR"), "/local_folds.rs"));
}

/// Restrict folding to operations whose effects are modeled by the evaluator.
/// A potentially trapping operation may only be erased after a successful fold.
pub(crate) fn can_reduce(dfg: &veloc_mir::dfg::DataFlowGraph, inst: veloc_mir::Inst) -> bool {
    let view = dfg.inst(inst);
    let results = dfg.inst_results(inst);
    !results.is_empty()
        && (local::can_fold(view.opcode())
            || (results
                .iter()
                .all(|&value| ScalarConst::from_bits(dfg.value_type(value), 0).is_some())
                && can_fold(view.opcode())))
        && view.memory_effect().is_none()
        && !view.is_terminator()
        && !view.opcode().transfers_ownership()
}

/// Reduce an existing or proposed expression without constructing MIR nodes.
/// Operand indices refer to the original argument order; callers may supply
/// canonical values for equality checks while retaining their own SSA witnesses.
pub(crate) fn reduce(
    opcode: Opcode,
    args: &[Value],
    results: &[Type],
    properties: &[IntCC],
    mut constant: impl FnMut(Value) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    if opcode == Opcode::Icmp
        && args.len() == 2
        && args[0] == args[1]
        && let [kind] = properties
    {
        let value = ScalarConst::from_bits(Type::BOOL, u64::from(kind.test(64, 0, 0))).unwrap();
        return Some(smallvec::smallvec![Fold::Constant(value)]);
    }
    if let [ty] = results
        && properties.is_empty()
        && let Some(fold) = local::fold(opcode, *ty, args, &mut constant)
    {
        return Some(smallvec::smallvec![fold]);
    }
    let constants: smallvec::SmallVec<[ScalarConst; 3]> = args
        .iter()
        .map(|&value| constant(value))
        .collect::<Option<_>>()?;
    evaluate(opcode, &constants, results, properties)
        .map(|values| values.into_iter().map(Fold::Constant).collect())
}
