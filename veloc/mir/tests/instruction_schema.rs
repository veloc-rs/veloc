use veloc_mir::dfg::DataFlowGraph;
use veloc_mir::inst::OpFormat;
use veloc_mir::inst::VectorExtData;
use veloc_mir::{Arguments, BlockCall, InstView};
use veloc_mir::{
    Block, CallConv, InstDraft, Linkage, MemFlags, ModuleBuilder, Opcode, Type, Value,
    VectorMemOptions,
};

fn operands(data: &InstView<'_>, include_auxiliary: bool) -> Vec<Value> {
    let mut values = Vec::new();
    if include_auxiliary {
        data.visit_operands(|value| values.push(value));
    } else {
        data.visit_type_operands(|value| values.push(value));
    }
    values
}

#[test]
fn rewriting_predicated_operands_preserves_construction_data() {
    let mut dfg = DataFlowGraph::new();
    let old = Value(0);
    let evl = Value(1);
    let new = Value(2);
    let args = Arguments::from_slice(&[old, old]);
    let ext = VectorExtData {
        mask: old,
        evl: Some(evl),
    };
    let original = InstDraft::vector_op_with_ext(Opcode::IAdd, &args, ext);
    let inst = dfg.create_inst(original.clone());
    dfg.replace_all_uses(old, new);
    let changed = dfg.inst(inst);

    assert_eq!(operands(&changed, false), [new, new]);
    assert_eq!(operands(&changed, true), [new, new, new, evl]);
    assert_eq!(operands(&original.as_view(), true), [old, old, old, evl]);
    assert!(changed.matches_format(OpFormat::Binary));
    assert!(!changed.matches_format(OpFormat::Unary));
    assert!(!changed.matches_format(OpFormat::IntCompare));
}

#[test]
fn branch_table_rewriting_visits_successor_arguments() {
    let mut dfg = DataFlowGraph::new();
    let old = Value(0);
    let keep = Value(1);
    let new = Value(2);
    let left_args = Arguments::from_slice(&[old, keep]);
    let right_args = Arguments::from_slice(&[keep, old]);
    let left = BlockCall {
        block: Block(0),
        args: left_args,
    };
    let right = BlockCall {
        block: Block(1),
        args: right_args,
    };
    let table = [left, right];
    let branch = InstDraft::br_table(old, table.iter().map(BlockCall::as_view));
    assert_eq!(
        operands(&branch.as_view(), false),
        [old, old, keep, keep, old]
    );
    let inst = dfg.create_inst(branch);
    dfg.replace_all_uses(old, new);
    let branch = dfg.draft(inst);
    assert_eq!(
        operands(&branch.as_view(), true),
        [new, new, keep, keep, new]
    );
}

#[test]
fn memory_view_keeps_auxiliary_operands_separate() {
    let values = [Value(0), Value(1), Value(2)];
    let flags = MemFlags::new().with_volatile(true).with_alignment(8);
    let ext = VectorMemOptions {
        flags,
        mask: Some(Value(3)),
        evl: Some(Value(4)),
        offset: 0,
        scale: 1,
    };
    let scatter = InstDraft::vector_scatter(values, ext);
    assert!(scatter.as_view().matches_format(OpFormat::VectorScatter));
    assert_eq!(operands(&scatter.as_view(), false), values);
    assert_eq!(
        operands(&scatter.as_view(), true),
        [Value(0), Value(1), Value(2), Value(3), Value(4)]
    );
    assert_eq!(scatter.as_view().memory_flags(), Some(flags));
    assert!(scatter.as_view().has_volatile_access());

    // Fixed operand groups cannot have the wrong length in construction data.
}

#[test]
fn values_construction_handles_inline_fixed_opcode_and_nullary_layouts() {
    let values = [Value(0), Value(1), Value(2)];
    for opcode in [
        Opcode::INeg,
        Opcode::IAdd,
        Opcode::Select,
        Opcode::IntToPtr,
        Opcode::PtrToInt,
        Opcode::Nop,
        Opcode::Unreachable,
    ] {
        let format = opcode.spec().format;
        let arity = format.fixed_value_arity().unwrap();
        let instruction = InstDraft::from_values(opcode, &values[..arity]).unwrap();
        assert_eq!(instruction.opcode(), opcode);
        assert!(instruction.as_view().matches_format(format));
        assert_eq!(operands(&instruction.as_view(), true), values[..arity]);
        assert_eq!(instruction.as_view().memory_flags(), None);
        assert!(InstDraft::from_values(opcode, &[Value(0); 4]).is_none());
    }
    // Property-bearing instructions cannot be fabricated from operands alone.
    assert!(InstDraft::from_values(Opcode::Load, &values[..1]).is_none());
}

#[test]
fn generated_memory_builders_preserve_field_order() {
    let mut module = ModuleBuilder::new();
    let signature = module.make_signature(vec![Type::PTR, Type::PTR], vec![], CallConv::SystemV);
    let function = module.declare_function("memory_fields".into(), signature, Linkage::Export);
    let mut builder = module.builder(function);
    builder.init_entry_block();
    let ptr = builder.func_param(0);
    let value = builder.func_param(1);
    let flags = MemFlags::new().with_volatile(true).with_alignment(8);

    // Both operands are pointers: reversing them would still pass type validation.
    builder.ins().store(ptr, value, 16, flags);
    let loaded = builder.ins().load(ptr, 16, flags, Type::PTR);

    let dfg = builder.func().dfg();
    let instructions: Vec<_> = dfg.instructions().map(|(_, data)| data).collect();
    assert!(matches!(
        instructions[0],
        InstView::Store { ptr: actual_ptr, value: actual_value, offset: 16, flags: actual_flags }
            if (actual_ptr, actual_value, actual_flags) == (ptr, value, flags)
    ));
    assert!(matches!(
        dfg.inst(dfg.value_inst(loaded).unwrap()),
        InstView::Load { ptr: actual_ptr, offset: 16, flags: actual_flags }
            if (actual_ptr, actual_flags) == (ptr, flags)
    ));
    assert_eq!(dfg.value_type(loaded), Type::PTR);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    module.validate().unwrap();
}

#[test]
fn generated_integer_constant_builder_preserves_bit_patterns() {
    let mut module = ModuleBuilder::new();
    let signature = module.make_signature(vec![], vec![], CallConv::SystemV);
    let function = module.declare_function("constant_bits".into(), signature, Linkage::Export);
    let mut builder = module.builder(function);
    builder.init_entry_block();

    let bits = 0xfedc_ba98_7654_3210;
    let raw = builder
        .ins()
        .iconst(veloc_mir::Int::from_bits(Type::I64, bits).expect("integer constant type"));
    let negative = builder.ins().i32const(-1);
    let minimum = builder.ins().i64const(i64::MIN);
    let dfg = builder.func().dfg();
    for (result, expected_bits, expected_type) in [
        (raw, bits, Type::I64),
        (negative, u32::MAX as u64, Type::I32),
        (minimum, 1 << 63, Type::I64),
    ] {
        assert!(matches!(
            dfg.inst(dfg.value_inst(result).unwrap()),
            InstView::Iconst { value } if value.to_bits() == expected_bits
        ));
        assert_eq!(dfg.value_type(result), expected_type);
    }

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    module.validate().unwrap();
}

#[test]
fn constants_share_scalar_storage_and_materialize_vectors() {
    use veloc_mir::{ConstData, Constant, Float, Int, ScalarConst, VectorConst};
    assert_eq!(size_of::<Int>(), size_of::<ScalarConst>());
    assert_eq!(size_of::<Float>(), size_of::<ScalarConst>());
    for (ty, width) in [
        (Type::I8, 8),
        (Type::I16, 16),
        (Type::I32, 32),
        (Type::I64, 64),
    ] {
        let value = Int::from_bits(ty, u64::MAX).unwrap();
        assert_eq!(value.signed(), -1);
        assert_eq!(value.to_bits(), u64::MAX >> (64 - width));
        assert_eq!(Int::try_from(ScalarConst::from(value)), Ok(value));
        assert!(Float::try_from(ScalarConst::from(value)).is_err());
    }
    assert!(ScalarConst::from_bits(Type::BOOL, 2).is_none());
    assert!(ScalarConst::from_bits(Type::PTR, 0).is_none());
    assert!(Int::from_bits(Type::F32, 0).is_none());
    let nan = Float::from_f32_bits(0x7fa12345);
    assert_eq!(ScalarConst::from(nan), ScalarConst::from(nan));
    assert_ne!(ScalarConst::from(0.0f32), ScalarConst::from(-0.0f32));
    assert_ne!(ScalarConst::from(0.0f32), ScalarConst::from(0.0f64));

    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
    let func = module.declare_function("constants".into(), sig, Linkage::Local);
    let mut builder = module.builder(func);
    builder.init_entry_block();
    let dfg = &mut DataFlowGraph::new();
    let bytes: Vec<_> = [1i32, -2, 3, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let make_dense = |ty: Type, bytes: Vec<u8>, dfg: &mut DataFlowGraph| {
        VectorConst::dense(
            ty.as_vector().unwrap(),
            veloc_mir::inst::ConstantPoolId::insert(dfg, bytes),
        )
    };
    let dense = make_dense(Type::I32X4, bytes.clone(), dfg);
    dense.validate(dfg).unwrap();
    assert!(matches!(dense.data(), ConstData::Dense(id) if id.get(dfg) == Some(bytes.as_slice())));
    assert_eq!(make_dense(Type::I32X4, bytes.clone(), dfg), dense);
    assert!(
        make_dense(Type::I32X4, vec![0; 3], dfg)
            .validate(dfg)
            .is_err()
    );
    let scalable = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, true)
        .unwrap()
        .as_type();
    assert!(
        make_dense(scalable, vec![0; 16], dfg)
            .validate(dfg)
            .is_err()
    );
    let mask = Type::new_mask(4, false).unwrap();
    assert!(
        make_dense(mask, vec![0, 1, 2, 0], dfg)
            .validate(dfg)
            .is_err()
    );
    let splat = VectorConst::splat(ScalarConst::from(-7i32), 4, false).unwrap();
    let scalable_splat = VectorConst::splat(ScalarConst::from(-7i32), 4, true).unwrap();
    assert_eq!(splat.ty(), Type::I32X4);
    assert!(VectorConst::splat(ScalarConst::from(7i32), 3, false).is_none());
    let dense = builder
        .func_mut()
        .edit()
        .dense_constant(Type::I32X4.as_vector().unwrap(), bytes);
    for value in [
        Constant::from(nan),
        dense.into(),
        splat.into(),
        scalable_splat.into(),
    ] {
        let result = builder.ins().constant(value);
        assert_eq!(builder.func().dfg().as_const(result), Some(value));
        assert_eq!(builder.func().dfg().value_type(result), value.ty());
        if value.as_vector().is_some() {
            assert!(matches!(
                builder
                    .func()
                    .dfg()
                    .inst(builder.func().dfg().value_inst(result).unwrap()),
                InstView::Vconst { .. }
            ));
        }
    }
    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    module.validate().unwrap();
    let text = module.build().to_string();
    let parsed = veloc_mir::ModuleParser::new().parse(&text).unwrap();
    parsed.validate().unwrap();
    assert_eq!(parsed.to_string(), text);
}

#[test]
fn vector_constant_construction_defers_data_checks_to_validation() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
    let func = module.declare_function("bad_constant".into(), sig, Linkage::Local);
    let mut builder = module.builder(func);
    builder.init_entry_block();
    let value = builder
        .func_mut()
        .edit()
        .dense_constant(Type::I32X4.as_vector().unwrap(), vec![0; 3]);
    let result = builder.ins().vconst(value);
    assert_eq!(builder.func().dfg().value_type(result), Type::I32X4);
    assert_eq!(builder.func().dfg().as_const(result), Some(value.into()));
    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    assert!(module.validate().is_err());
}
