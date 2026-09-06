use veloc_mir::dfg::{DataFlowGraph, PoolKey};
use veloc_mir::inst::{VectorExtData, VectorExtId, VectorMemExtId};
use veloc_mir::opspec::OpFormat;
use veloc_mir::types::{BlockCallData, JumpTableData};
use veloc_mir::{
    Block, CallConv, InstructionData, Linkage, MemFlags, ModuleBuilder, Opcode, Type, Value,
    VectorMemOptions,
};

fn operands(data: &InstructionData, dfg: &DataFlowGraph, include_auxiliary: bool) -> Vec<Value> {
    let mut values = Vec::new();
    if include_auxiliary {
        data.visit_operands(dfg, |value| values.push(value));
    } else {
        data.visit_type_operands(dfg, |value| values.push(value));
    }
    values
}

#[test]
fn rewriting_predicated_operands_preserves_interned_original() {
    let mut dfg = DataFlowGraph::new();
    let old = Value(0);
    let evl = Value(1);
    let new = Value(2);
    let args = dfg.make_value_list(&[old, old]);
    let ext = VectorExtId::insert(
        &mut dfg,
        VectorExtData {
            mask: old,
            evl: Some(evl),
        },
    );
    let original = InstructionData::VectorOpWithExt {
        opcode: Opcode::IAdd,
        args,
        ext,
    };
    let mut changed = original.clone();
    changed.replace_value(&mut dfg, old, new);

    assert_eq!(operands(&changed, &dfg, false), [new, new]);
    assert_eq!(operands(&changed, &dfg, true), [new, new, new, evl]);
    assert_eq!(operands(&original, &dfg, true), [old, old, old, evl]);
    assert!(changed.matches_format(&dfg, OpFormat::Binary));
    assert!(!changed.matches_format(&dfg, OpFormat::Unary));
    assert!(!changed.matches_format(&dfg, OpFormat::IntCompare));
}

#[test]
fn branch_table_rewriting_visits_successor_arguments() {
    let mut dfg = DataFlowGraph::new();
    let old = Value(0);
    let keep = Value(1);
    let new = Value(2);
    let left_args = dfg.make_value_list(&[old, keep]);
    let right_args = dfg.make_value_list(&[keep, old]);
    let left = dfg.block_calls.push(BlockCallData {
        block: Block(0),
        args: left_args,
    });
    let right = dfg.block_calls.push(BlockCallData {
        block: Block(1),
        args: right_args,
    });
    let table = dfg.jump_tables.push(JumpTableData {
        targets: vec![left, right],
    });
    let mut branch = InstructionData::BrTable { index: old, table };
    assert_eq!(operands(&branch, &dfg, false), [old, old, keep, keep, old]);
    branch.replace_value(&mut dfg, old, new);
    assert_eq!(operands(&branch, &dfg, true), [new, new, keep, keep, new]);
}

#[test]
fn pooled_memory_layout_checks_arity_and_keeps_auxiliary_operands_separate() {
    let mut dfg = DataFlowGraph::new();
    let values = [Value(0), Value(1), Value(2)];
    let flags = MemFlags::VOLATILE.with_alignment(8);
    let ext = VectorMemExtId::insert(
        &mut dfg,
        VectorMemOptions {
            flags,
            mask: Some(Value(3)),
            evl: Some(Value(4)),
            ..VectorMemOptions::default()
        },
    );
    let args = dfg.make_value_list(&values);
    let scatter = InstructionData::VectorScatter { args, ext };
    assert!(scatter.matches_format(&dfg, OpFormat::VectorScatter));
    assert_eq!(operands(&scatter, &dfg, false), values);
    assert_eq!(
        operands(&scatter, &dfg, true),
        [Value(0), Value(1), Value(2), Value(3), Value(4)]
    );
    assert_eq!(scatter.memory_flags(&dfg), Some(flags));
    assert!(scatter.memory_effect(&dfg).volatile);

    let args = dfg.make_value_list(&values[..2]);
    let incomplete = InstructionData::VectorScatter { args, ext };
    assert!(!incomplete.matches_format(&dfg, OpFormat::VectorScatter));
}

#[test]
fn values_construction_handles_inline_fixed_opcode_and_nullary_layouts() {
    let dfg = DataFlowGraph::new();
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
        let instruction = InstructionData::from_values(opcode, &values[..arity]).unwrap();
        assert_eq!(instruction.opcode(), opcode);
        assert!(instruction.matches_format(&dfg, format));
        assert_eq!(operands(&instruction, &dfg, true), values[..arity]);
        assert_eq!(instruction.memory_flags(&dfg), None);
        assert!(InstructionData::from_values(opcode, &[Value(0); 4]).is_none());
    }
    // Property-bearing instructions cannot be fabricated from operands alone.
    assert!(InstructionData::from_values(Opcode::Load, &values[..1]).is_none());
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
    let flags = MemFlags::VOLATILE.with_alignment(8);

    // Both operands are pointers: reversing them would still pass type validation.
    builder.ins().store(ptr, value, 16, flags);
    let loaded = builder.ins().load(ptr, 16, flags, Type::PTR);
    builder.ins().stack_store(slot, value, 24);
    let stacked = builder.ins().stack_load(slot, 24, Type::PTR);

    let dfg = &builder.func().dfg;
    let instructions: Vec<_> = dfg.instructions.values().collect();
    assert!(matches!(
        instructions[0],
        InstructionData::Store { ptr: actual_ptr, value: actual_value, offset: 16, flags: actual_flags }
            if (*actual_ptr, *actual_value, *actual_flags) == (ptr, value, flags)
    ));
    assert!(matches!(
        dfg.inst(dfg.value_inst(loaded).unwrap()),
        InstructionData::Load { ptr: actual_ptr, offset: 16, flags: actual_flags }
            if (*actual_ptr, *actual_flags) == (ptr, flags)
    ));
    assert!(matches!(
        instructions[2],
        InstructionData::StackStore { slot: actual_slot, value: actual_value, offset: 24 }
            if (*actual_slot, *actual_value) == (slot, value)
    ));
    assert!(matches!(
        dfg.inst(dfg.value_inst(stacked).unwrap()),
        InstructionData::StackLoad { slot: actual_slot, offset: 24 } if *actual_slot == slot
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
    let dfg = &builder.func().dfg;
    for (result, expected_bits, expected_type) in [
        (raw, bits, Type::I64),
        (negative, u64::MAX, Type::I32),
        (minimum, 1 << 63, Type::I64),
    ] {
        assert!(matches!(
            dfg.inst(dfg.value_inst(result).unwrap()),
            InstructionData::Iconst { value } if *value == expected_bits
        ));
        assert_eq!(dfg.value_type(result), expected_type);
    }

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    module.validate().unwrap();
}
