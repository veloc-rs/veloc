use veloc_analyzer::{AnalysisManager, UseDefAnalysis};
use veloc_mir::constant::Constant;
use veloc_mir::{CallConv, InstructionData, Linkage, ModuleBuilder, Opcode, Type};
use veloc_optimizer::Metrics;
use veloc_optimizer::passes::function::constant_folding::run_constant_folding;

#[test]
fn a_nonconstant_operand_in_any_position_prevents_folding() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![Type::I32], vec![Type::I32; 2], CallConv::SystemV);
    let id = module.declare_function("partial_constants".into(), sig, Linkage::Local);
    {
        let mut builder = module.builder(id);
        builder.init_entry_block();
        let arg = builder.func_params()[0];
        let mut ins = builder.ins();
        let one = ins.i32const(1);
        let results = [ins.iadd(arg, one), ins.iadd(one, arg)];
        ins.ret(&results);
    }
    let mut data = module.build_data();
    data.validate().unwrap();
    let func = &mut data.functions[id];
    let revision = func.revision();
    assert!(!run_constant_folding(
        func,
        &mut AnalysisManager::new(),
        false,
        &mut Metrics::default()
    ));
    assert_eq!(func.revision(), revision);
    data.validate().unwrap();
}

#[test]
fn constant_traps_remain_instructions_while_safe_division_folds() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![], vec![Type::I32; 4], CallConv::SystemV);
    let id = module.declare_function("division".into(), sig, Linkage::Local);
    let values = {
        let mut builder = module.builder(id);
        builder.init_entry_block();
        let mut ins = builder.ins();
        let min = ins.i32const(i32::MIN);
        let zero = ins.i32const(0);
        let neg_one = ins.i32const(-1);
        let two = ins.i32const(2);
        let values = [
            ins.idiv_s(min, neg_one),
            ins.idiv_u(two, zero),
            ins.idiv_s(min, two),
            ins.irem_s(min, neg_one),
        ];
        ins.ret(&values);
        values
    };
    module.validate().unwrap();
    let mut data = module.build_data();
    let func = &mut data.functions[id];
    let mut analyses = AnalysisManager::new();
    let mut metrics = Metrics::default();
    analyses.use_def(func);
    assert!(run_constant_folding(
        func,
        &mut analyses,
        false,
        &mut metrics
    ));
    for (value, op) in values[..2].iter().zip([Opcode::IDivS, Opcode::IDivU]) {
        assert_eq!(func.dfg.as_const(*value), None);
        let veloc_mir::ValueDef::Inst(inst) = func.dfg.values[*value].def else {
            panic!("expected instruction result")
        };
        assert_eq!(func.dfg.inst(inst).opcode(), op);
    }
    assert_eq!(
        func.dfg.as_const(values[2]),
        Some(Constant::I32(i32::MIN / 2))
    );
    assert_eq!(func.dfg.as_const(values[3]), Some(Constant::I32(0)));
    let rebuilt = UseDefAnalysis::new(func);
    for value in func.dfg.values.keys() {
        assert_eq!(
            analyses.use_def(func).users_of(value),
            rebuilt.users_of(value)
        );
    }
    assert!(!run_constant_folding(
        func,
        &mut analyses,
        false,
        &mut metrics
    ));
    data.validate().unwrap();
}

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

#[test]
fn multi_result_folding_preserves_values_users_layout_and_types() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(
        vec![],
        vec![Type::I64, Type::BOOL, Type::BOOL, Type::I8],
        CallConv::SystemV,
    );
    let id = module.declare_function("typed_chain".into(), sig, Linkage::Local);
    let values = {
        let mut builder = module.builder(id);
        builder.init_entry_block();
        let mut ins = builder.ins();
        let x = ins.iconst(i8::MAX as u64, Type::I8);
        let y = ins.iconst(1, Type::I8);
        let (sum, overflow) = ins.iadd_with_overflow(x, y);
        let wide = ins.extends(sum, Type::I64);
        let negative = ins.icmp(veloc_mir::IntCC::LtS, sum, y);
        let narrow = ins.wrap(wide, Type::I8);
        let values = [wide, overflow, negative, narrow];
        ins.ret(&values);
        values
    };
    module.validate().unwrap();
    let mut data = module.build_data();
    let func = &mut data.functions[id];
    let value_count = func.dfg.values.len();
    let mut analyses = AnalysisManager::new();
    let mut metrics = Metrics::default();
    analyses.use_def(func);
    assert!(run_constant_folding(
        func,
        &mut analyses,
        false,
        &mut metrics
    ));
    assert_eq!(func.dfg.values.len(), value_count);
    for (value, expected) in values.into_iter().zip([
        Constant::I64(-128),
        Constant::Bool(true),
        Constant::Bool(true),
        Constant::I8(-128),
    ]) {
        assert_eq!(func.dfg.as_const(value), Some(expected));
    }
    let rebuilt = UseDefAnalysis::new(func);
    for value in func.dfg.values.keys() {
        assert_eq!(
            analyses.use_def(func).users_of(value),
            rebuilt.users_of(value)
        );
    }
    assert!(!run_constant_folding(
        func,
        &mut analyses,
        false,
        &mut metrics
    ));
    data.validate().unwrap();
}
