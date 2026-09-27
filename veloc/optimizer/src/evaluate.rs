//! Generated scalar evaluation consumed by the e-graph's constant analysis.
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
