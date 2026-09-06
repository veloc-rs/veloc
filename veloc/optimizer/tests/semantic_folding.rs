use veloc_analyzer::{AnalysisManager, UseDefAnalysis};
use veloc_mir::constant::Constant;
use veloc_mir::{CallConv, InstructionData, Linkage, ModuleBuilder, Opcode, Type};
use veloc_optimizer::Metrics;
use veloc_optimizer::passes::function::constant_folding::run_constant_folding;

#[test]
fn semantic_folding_propagates_wrapping_values_through_ssa_and_returns() {
    // The pass must execute the composite `0 - arg` contract, not require a
    // one-op identity or recover the former BvOp::Neg binding.
    assert_eq!(Opcode::INeg.spec().semantics.unwrap().primitive(), None);
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![], vec![Type::I32; 7], CallConv::SystemV);
    let func_id = module.declare_function("wrapping_chain".into(), sig, Linkage::Local);
    let (entry, values) = {
        let mut builder = module.builder(func_id);
        let entry = builder.init_entry_block();
        let mut ins = builder.ins();
        let max = ins.i32const(i32::MAX);
        let one = ins.i32const(1);
        let minus_one = ins.i32const(-1);

        let sum = ins.iadd(max, one);
        let flipped = ins.ixor(sum, minus_one);
        let negated = ins.ineg(flipped);
        let difference = ins.isub(negated, one);
        let product = ins.imul(difference, minus_one);
        let masked = ins.iand(product, minus_one);
        let merged = ins.ior(masked, one);
        let values = [sum, flipped, negated, difference, product, masked, merged];
        ins.ret(&values);
        (entry, values)
    };
    module.validate().unwrap();
    let mut data = module.build_data();
    let func = &mut data.functions[func_id];
    let ret = *func.layout.blocks[entry].insts.last().unwrap();
    let mut analyses = AnalysisManager::new();
    let mut metrics = Metrics::default();

    // Populate the cache before folding so the test also exercises updates to
    // existing use-def information as the arithmetic operands disappear.
    assert_eq!(analyses.use_def(func).users_of(values[0]).len(), 2);
    assert!(run_constant_folding(
        func,
        &mut analyses,
        false,
        &mut metrics
    ));

    for (value, expected) in values.into_iter().zip([
        i32::MIN,
        i32::MAX,
        -i32::MAX,
        i32::MIN,
        i32::MIN,
        i32::MIN,
        i32::MIN + 1,
    ]) {
        assert_eq!(func.dfg.as_const(value), Some(Constant::I32(expected)));
        assert_eq!(func.dfg.value_type(value), Type::I32);
    }

    // Folding changes definitions in place; the return keeps its SSA Value IDs
    // and now observes the folded constants without a separate RAUW pass.
    let InstructionData::Return { values: returned } = func.dfg.inst(ret) else {
        panic!("folding replaced the return instruction");
    };
    assert_eq!(func.dfg.get_value_list(*returned), values);
    assert_eq!(
        metrics.counters.get("constant_folding.folded_insts"),
        Some(&7)
    );

    let rebuilt = UseDefAnalysis::new(func);
    let cached = analyses.use_def(func);
    for value in func.dfg.values.keys() {
        assert_eq!(cached.users_of(value), rebuilt.users_of(value));
    }

    let revision = func.revision();
    assert!(!run_constant_folding(
        func,
        &mut analyses,
        false,
        &mut metrics
    ));
    assert_eq!(func.revision(), revision);
    assert_eq!(
        metrics.counters.get("constant_folding.folded_insts"),
        Some(&7)
    );
    data.validate().unwrap();
}
