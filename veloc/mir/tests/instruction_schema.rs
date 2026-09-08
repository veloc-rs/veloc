use veloc_mir::dfg::DataFlowGraph;
use veloc_mir::inst::OpFormat;
use veloc_mir::inst::VectorExtData;
use veloc_mir::{Arguments, BlockCall, InstructionView};
use veloc_mir::{
    Block, CallConv, InstDraft, Linkage, MemFlags, ModuleBuilder, Opcode, Type, Value,
    VectorMemOptions,
};

fn operands(data: &InstructionView<'_>, include_auxiliary: bool) -> Vec<Value> {
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
        ..VectorMemOptions::default()
    };
    let scatter = InstDraft::vector_scatter(values, ext);
    assert!(scatter.as_view().matches_format(OpFormat::VectorScatter));
    assert_eq!(operands(&scatter.as_view(), false), values);
    assert_eq!(
        operands(&scatter.as_view(), true),
        [Value(0), Value(1), Value(2), Value(3), Value(4)]
    );
    assert_eq!(scatter.as_view().memory_flags(), Some(flags));
    assert!(scatter.as_view().memory_effect().volatile);

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
    let slot = builder.create_stack_slot(64);
    let flags = MemFlags::new().with_volatile(true).with_alignment(8);

    // Both operands are pointers: reversing them would still pass type validation.
    builder.ins().store(ptr, value, 16, flags);
    let loaded = builder.ins().load(ptr, 16, flags, Type::PTR);
    builder.ins().stack_store(slot, value, 24);
    let stacked = builder.ins().stack_load(slot, 24, Type::PTR);

    let dfg = builder.func().dfg();
    let instructions: Vec<_> = dfg.instructions().map(|(_, data)| data).collect();
    assert!(matches!(
        instructions[0],
        InstructionView::Store { ptr: actual_ptr, value: actual_value, offset: 16, flags: actual_flags }
            if (actual_ptr, actual_value, actual_flags) == (ptr, value, flags)
    ));
    assert!(matches!(
        dfg.inst(dfg.value_inst(loaded).unwrap()),
        InstructionView::Load { ptr: actual_ptr, offset: 16, flags: actual_flags }
            if (actual_ptr, actual_flags) == (ptr, flags)
    ));
    assert!(matches!(
        instructions[2],
        InstructionView::StackStore { slot: actual_slot, value: actual_value, offset: 24 }
            if (actual_slot, actual_value) == (slot, value)
    ));
    assert!(matches!(
        dfg.inst(dfg.value_inst(stacked).unwrap()),
        InstructionView::StackLoad { slot: actual_slot, offset: 24 } if actual_slot == slot
    ));
    assert_eq!(dfg.value_type(loaded), Type::PTR);
    assert_eq!(dfg.value_type(stacked), Type::PTR);

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
    let raw = builder.ins().iconst(bits, Type::I64);
    let negative = builder.ins().i32const(-1);
    let minimum = builder.ins().i64const(i64::MIN);
    let dfg = builder.func().dfg();
    for (result, expected_bits, expected_type) in [
        (raw, bits, Type::I64),
        (negative, u64::MAX, Type::I32),
        (minimum, 1 << 63, Type::I64),
    ] {
        assert!(matches!(
            dfg.inst(dfg.value_inst(result).unwrap()),
            InstructionView::Iconst { value } if value == expected_bits
        ));
        assert_eq!(dfg.value_type(result), expected_type);
    }

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    module.validate().unwrap();
}
