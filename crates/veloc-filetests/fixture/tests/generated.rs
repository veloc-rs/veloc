//! Compile and execute generated APIs against the real MIR implementation.
//! File tests cover text/diagnostics; these cover APIs text cannot exercise.
extern crate alloc;
extern crate veloc_test_mir as veloc_mir;

use veloc_mir::constant::ScalarConst;
use veloc_mir::{
    Arguments, BlockCall, CallConv, InstDraft, InstructionView, IntCC, Linkage, ModuleBuilder,
    Opcode, Type, Value,
};

#[allow(dead_code)]
mod evaluator {
    use super::*;
    enum Replacement {
        Constants(Vec<ScalarConst>),
        Value(Value),
    }
    include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));
}
mod offline {
    use veloc_mir::{IntCC, Opcode};
    include!(concat!(env!("OUT_DIR"), "/semantics.rs"));
}
include!(concat!(env!("OUT_DIR"), "/lowering.rs"));

#[test]
fn new_ops_get_constraints_and_ownership_without_rust_opcode_cases() {
    let parse = |text: &str| veloc_mir::ModuleParser::new().parse(text).unwrap();
    let valid = "local function test(owned<() -> void>) -> owned<() -> void>\nblock0(v0: owned<() -> void>):\n  v1: owned<() -> void> = rebind v0\n  return v1\n";
    parse(valid).validate().unwrap();
    let twice = valid.replace("return v1", "closure-drop v0\n  return v1");
    assert!(
        parse(&twice)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("used after move")
    );
    let changed = valid.replace("v1: owned<() -> void>", "v1: shared<() -> void>");
    assert!(
        parse(&changed)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("rebind type mismatch")
    );
    let observe = valid.replace(
        "v1: owned<() -> void> = rebind v0",
        "observe v0\n  v1: owned<() -> void> = rebind v0",
    );
    assert!(
        parse(&observe)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("no ownership transfer contract")
    );

    let branch = "local function test(owned<() -> void>, owned<() -> void>) -> void\nblock0(v0: owned<() -> void>, v1: owned<() -> void>):\n  move-branch v0, block1(v1)\nblock1(v2: owned<() -> void>):\n  closure-drop v2\n  return\n";
    parse(branch).validate().unwrap();
    let twice = branch.replace("block1(v1)", "block1(v0)");
    assert!(
        parse(&twice)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("used after move")
    );
}

#[test]
fn definition_owned_records_flatten_operands_in_field_order() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::TestOperands;
    let mut dfg = DataFlowGraph::new();
    for optional in [None, Some(Value(1))] {
        let inst = dfg.create_inst(InstDraft::grouped(
            Opcode::Grouped,
            Value(0),
            TestOperands {
                token: Value(1),
                optional,
                tag: 17,
            },
            Value(2),
        ));
        let expected = if optional.is_some() {
            vec![Value(0), Value(1), Value(1), Value(2)]
        } else {
            vec![Value(0), Value(1), Value(2)]
        };
        assert_eq!(dfg.operands(inst), expected);
        // Errors in auxiliary record operands must stop traversal too, not
        // merely suppress later callbacks or skip optional fields.
        for stop in 0..expected.len() {
            let mut visited = Vec::new();
            let result = dfg.inst(inst).try_visit_operands(|value| {
                visited.push(value);
                if visited.len() == stop + 1 {
                    Err(stop)
                } else {
                    Ok(())
                }
            });
            assert_eq!(result, Err(stop));
            assert_eq!(visited, expected[..=stop]);
        }
        dfg.set_operand(inst, 1, Value(3));
        let InstructionView::Grouped {
            before,
            group,
            after,
            ..
        } = dfg.inst(inst)
        else {
            unreachable!()
        };
        assert_eq!(
            (before, group.token, group.optional, group.tag, after),
            (Value(0), Value(3), optional, 17, Value(2))
        );
        let mut primary = vec![];
        dfg.inst(inst).visit_type_operands(|v| primary.push(v));
        assert_eq!(primary, [Value(0), Value(2)]);
        dfg.check_uses().unwrap();
    }
}

#[test]
fn variadic_ranges_grow_recycle_and_remain_independent_after_clone() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let left = dfg.create_inst(InstDraft::nop());
    let right = dfg.create_inst(InstDraft::binary(Opcode::IAdd, [Value(7), Value(7)]));
    for size in [0, 1, 2, 3, 4, 5, 17, 65, 3, 0, 65, 17, 5, 1] {
        let args: Arguments = (0..size).map(|i| Value(i % 7)).collect();
        dfg.replace_inst(left, InstDraft::ret(&args));
        assert_eq!(dfg.operands(left), args.as_slice());
        assert_eq!(dfg.operands(right), &[Value(7), Value(7)]);
        let InstructionView::Return { values } = dfg.inst(left) else {
            unreachable!()
        };
        assert_eq!(values.as_ptr(), dfg.operands(left).as_ptr());
        dfg.check_uses().unwrap();
    }
    let mut copy = dfg.clone();
    copy.replace_all_uses(Value(7), Value(8));
    assert_eq!(dfg.operands(right), &[Value(7), Value(7)]);
    assert_eq!(copy.operands(right), &[Value(8), Value(8)]);
    dfg.check_uses().unwrap();
    copy.check_uses().unwrap();
}

#[test]
fn function_edits_keep_layout_and_successor_edges_in_sync() {
    let module = veloc_mir::ModuleParser::new().parse("local function main() -> void\nblock0():\n  jump block1()\nblock1():\n  return\nblock2():\n  return\n").unwrap();
    let mut data = (*module).clone();
    let (_, func) = data.functions.iter_mut().next().unwrap();
    let entry = func.entry_block.unwrap();
    let old = func.layout().block_order()[1];
    let new = func.layout().block_order()[2];
    let jump = func.layout().blocks()[entry].insts[0];
    let dest = BlockCall::new(new, &[]);
    func.edit()
        .replace_inst(jump, InstDraft::jump(dest.as_view()));
    assert!(func.layout().blocks()[old].preds.is_empty());
    assert_eq!(func.layout().blocks()[new].preds, [entry]);
    assert_eq!(func.layout().blocks()[entry].succs, [new]);
    func.edit().erase_inst(jump);
    assert!(func.layout().inst_block(jump).is_none());
    assert!(func.layout().blocks()[new].preds.is_empty());
    assert!(func.layout().blocks()[entry].succs.is_empty());
    let replacement = func
        .edit()
        .append_inst(entry, InstDraft::jump(dest.as_view()), &[]);
    assert_eq!(func.layout().inst_block(replacement), Some(entry));
    func.dfg().check_uses().unwrap();
    data.validate().unwrap();
}

#[test]
fn borrowed_uses_distinguish_operands_and_edits_update_the_single_storage() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let inst = dfg.create_inst(InstDraft::binary(Opcode::IAdd, [Value(0), Value(0)]));
    assert_eq!(dfg.uses(Value(0)).count(), 2);
    assert!(!dfg.has_one_use(Value(0)));
    let mut positions = dfg
        .uses(Value(0))
        .map(|site| (site.inst(), site.index(), site.value()))
        .collect::<Vec<_>>();
    positions.sort_unstable_by_key(|site| site.1);
    assert_eq!(positions, [(inst, 0, Value(0)), (inst, 1, Value(0))]);
    dfg.set_operand(inst, 0, Value(1));
    assert_eq!(dfg.operands(inst), &[Value(1), Value(0)]);
    assert!(dfg.has_one_use(Value(0)));
    dfg.replace_all_uses(Value(0), Value(1));
    dfg.replace_all_uses(Value(1), Value(1));
    assert!(dfg.use_empty(Value(0)));
    assert_eq!(dfg.uses(Value(1)).count(), 2);
    dfg.check_uses().unwrap();
    let before = dfg.clone();
    dfg.replace_inst(inst, InstDraft::unary(Opcode::INeg, Value(2)));
    let invalid = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dfg.set_operand(inst, 1, Value(3))
    }));
    assert!(invalid.is_err());
    assert_eq!(before.uses(Value(1)).count(), 2);
    assert!(dfg.use_empty(Value(1)));
    dfg.check_uses().unwrap();
}

#[test]
fn unified_slots_distinguish_repeated_successors_and_vector_operands() {
    use veloc_mir::Block;
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::VectorExtData;
    let mut dfg = DataFlowGraph::new();
    let call = BlockCall::new(Block(0), &[Value(0)]);
    let table = [call.clone(), call.clone()];
    let data = InstDraft::br_table(Value(0), table.iter().map(BlockCall::as_view));
    let first = dfg.create_inst(data.clone());
    let other = dfg.create_inst(data);
    dfg.set_operand(first, 1, Value(1));
    assert_eq!(dfg.operands(first)[2], Value(0));
    assert_eq!(dfg.operands(other)[1], Value(0));
    assert_eq!(call.args.as_slice(), &[Value(0)]);
    let ext = VectorExtData {
        mask: Value(0),
        evl: Some(Value(0)),
    };
    let args = Arguments::from_slice(&[Value(0), Value(0)]);
    let data = InstDraft::vector_op_with_ext(Opcode::IAdd, &args, ext);
    let first = dfg.create_inst(data.clone());
    let other = dfg.create_inst(data);
    dfg.set_operand(first, 2, Value(2));
    assert_eq!(dfg.operands(first)[3], Value(0));
    assert_eq!(dfg.operands(other)[2], Value(0));
    assert_eq!(ext.mask, Value(0));
    dfg.replace_all_uses(Value(0), Value(3));
    dfg.check_uses().unwrap();
}

#[test]
fn exact_use_index_survives_deterministic_edit_sequences() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let insts: Vec<_> = (0..32).map(|_| dfg.create_inst(InstDraft::nop())).collect();
    let mut random = 12345u32;
    for _ in 0..1000 {
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        let inst = insts[(random >> 16) as usize % insts.len()];
        let value = Value((random >> 8) % 8);
        match random % 4 {
            0 => dfg.replace_inst(inst, InstDraft::binary(Opcode::IAdd, [value, value])),
            1 => dfg.replace_inst(inst, InstDraft::unary(Opcode::INeg, value)),
            2 => dfg.remove_inst(inst),
            _ => dfg.replace_all_uses(value, Value((value.0 + 1) % 8)),
        }
        dfg.check_uses().unwrap();
    }
}

#[test]
fn closed_dead_cycles_are_erased_together() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let a = dfg.create_inst(InstDraft::unary(Opcode::INeg, Value(1)));
    dfg.append_results(a, &[Type::I32]);
    let b = dfg.create_inst(InstDraft::unary(Opcode::INeg, Value(0)));
    dfg.append_results(b, &[Type::I32]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dfg.remove_inst(a))).is_err());
    dfg.check_uses().unwrap();
    dfg.remove_insts(&[a, b]);
    assert!(dfg.use_empty(Value(0)) && dfg.use_empty(Value(1)));
    dfg.check_uses().unwrap();
}

#[test]
fn generated_encodings_preserve_neighboring_fields_and_check_ranges() {
    use veloc_mir::inst::{MemFlags, PackedRecord, WideRecord};
    const PACKED: PackedRecord = PackedRecord::empty()
        .with_enabled(true)
        .with_count(7)
        .with_tag(15);
    assert!(PACKED.is_enabled());
    assert_eq!(PACKED.count(), 7);
    assert_eq!(PACKED.tag(), 15);
    let cleared = PACKED.with_count(0).with_enabled(false);
    assert!(!cleared.is_enabled());
    assert_eq!(cleared.count(), 0);
    assert_eq!(cleared.tag(), 15);
    assert_eq!(WideRecord::empty().with_value(u128::MAX).value(), u128::MAX);
    assert!(std::panic::catch_unwind(|| PACKED.with_count(8)).is_err());
    assert_eq!(MemFlags::default().alignment(), 1);
    for log2 in 0..32 {
        let flags = MemFlags::new()
            .with_volatile(true)
            .with_alignment(1 << log2);
        assert!(flags.is_volatile());
        assert_eq!(flags.alignment(), 1 << log2.min(15));
        assert_eq!(flags.with_volatile(false).alignment(), flags.alignment());
    }
    for invalid in [0, 3, u32::MAX] {
        assert!(std::panic::catch_unwind(|| MemFlags::new().with_alignment(invalid)).is_err());
    }
}

#[test]
fn scalar_enum_is_exhaustive_and_preserves_type_encoding() {
    use veloc_mir::{CallableKind, ScalarType, SigId};

    // No wildcard: adding a scalar kind must prompt consumers to consider it.
    fn width(ty: ScalarType) -> Option<u32> {
        match ty {
            ScalarType::I8 => Some(8),
            ScalarType::I16 => Some(16),
            ScalarType::I32 | ScalarType::F32 => Some(32),
            ScalarType::I64 | ScalarType::F64 => Some(64),
            ScalarType::BOOL => Some(1),
            ScalarType::PTR => None,
        }
    }

    assert_eq!(size_of::<ScalarType>(), 1);
    assert_eq!(size_of::<Option<ScalarType>>(), 1);
    assert_eq!(size_of::<Type>(), 8);
    const SCALAR: ScalarType = Type::I32.as_scalar().unwrap();
    const VECTOR: Type = SCALAR.vector(4, false).unwrap().as_type();
    assert_eq!(SCALAR, ScalarType::I32);
    assert_eq!(VECTOR, Type::I32X4);
    for code in 0..=u8::MAX {
        let scalar = ScalarType::from_code(code);
        assert_eq!(
            Type::from_scalar_code(code),
            scalar.map(ScalarType::as_type)
        );
        if let Some(scalar) = scalar {
            assert_eq!(scalar.code(), code);
            assert_eq!(scalar as u8, code);
            assert_eq!(scalar.as_type().as_scalar(), Some(scalar));
            assert_eq!(scalar.as_type().element_bits(), width(scalar));
            for scalable in [false, true] {
                if let Some(vector) = scalar.vector(4, scalable) {
                    assert_eq!(vector.element_type(), scalar);
                    assert_eq!(vector.as_type().as_scalar(), None);
                } else {
                    assert_eq!(scalar, ScalarType::PTR);
                }
            }
        }
    }
    assert_eq!(ScalarType::from_code(0), None);
    assert_eq!(Type::INVALID.as_scalar(), None);
    for kind in [
        CallableKind::Local,
        CallableKind::Owned,
        CallableKind::Shared,
    ] {
        assert_eq!(Type::callable(SigId(0), kind).as_scalar(), None);
    }
    for raw in 0..=u16::MAX {
        if let Some(ty) = Type::from_raw(raw) {
            assert_eq!(ty.as_scalar().is_some(), ty.as_vector().is_none());
        }
    }
}

#[test]
fn generated_flag_sets_preserve_bits_order_and_set_operations() {
    use veloc_mir::inst::{EmptyFlags, MemoryRegions, OpTraits, TestFlags as F};
    const SELECTED: F = F::HIGH.union(F::LOW_BIT);
    const {
        assert!(F::ALL.contains(SELECTED));
        assert!(SELECTED.contains(F::HIGH));
        assert!(!SELECTED.contains(F::ALL));
        assert!(SELECTED.intersects(F::LOW_BIT.union(F::MIDDLE)));
        assert!(!SELECTED.intersects(F::MIDDLE));
        assert!(F::NONE.is_empty());
        assert!(F::HIGH.contains(F::NONE));
        assert!(!F::HIGH.intersects(F::NONE));
        assert!(EmptyFlags::ALL.is_empty());
    }
    assert_eq!(size_of::<F>(), 16);
    assert_eq!(size_of::<MemoryRegions>(), 1);
    assert_eq!(size_of::<OpTraits>(), 2);
    assert_eq!(F::HIGH.union(F::LOW_BIT).union(F::MIDDLE), F::ALL);
    assert_eq!(F::ALL.to_string(), "high / low-bit / middle");
    assert_eq!(SELECTED.to_string(), "high / low-bit");
    assert_eq!(F::empty().to_string(), "none");
    assert_eq!(EmptyFlags::ALL.to_string(), "none");
    assert_eq!(
        MemoryRegions::HEAP.union(MemoryRegions::STACK).to_string(),
        "heap,stack"
    );
    assert_eq!(
        OpTraits::TERMINATOR
            .union(OpTraits::COMMUTATIVE)
            .to_string(),
        "terminator, commutative"
    );
}

#[test]
fn construction_does_not_validate_type_contracts() {
    for (case, expected) in [
        ("binding", "Pattern { results: false, index: 1"),
        ("class", "Pattern { results: false, index: 0"),
        ("explicit", "Pattern { results: true, index: 0"),
        ("float-type", "result 0 must have the type of `value`"),
        ("relation", "results[0] must have more bits"),
        ("fixed", "Pattern { results: false, index: 0"),
        ("raw-results", "Pattern { results: true, index: 0"),
        ("raw-arity", "Arity { results: true"),
        ("call", "call value 0 type mismatch"),
        ("indirect-call", "Pattern { results: false, index: 0"),
        ("branch", "Pattern { results: false, index: 0"),
        ("table", "Pattern { results: false, index: 0"),
    ] {
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
        let id = module.declare_function(case.into(), sig, Linkage::Local);
        let callee_sig = module.make_signature(
            vec![Type::F32],
            vec![Type::I64, Type::I32],
            CallConv::SystemV,
        );
        let callee = module.declare_function("callee".into(), callee_sig, Linkage::Import);
        let mut builder = module.builder(id);
        builder.init_entry_block();
        let mut ins = builder.ins();
        let i = ins.i32const(1);
        let f = ins.f32const(1.0);
        match case {
            "binding" => {
                let result = ins.first(i, f);
                assert_eq!(ins.value_type(result), Type::I32);
            }
            "class" => {
                let result = ins.float_only(i, i);
                assert_eq!(ins.value_type(result), Type::I32);
            }
            "explicit" => {
                let result = ins.output(Type::F32);
                assert_eq!(ins.value_type(result), Type::F32);
            }
            "float-type" => {
                let data = InstDraft::fconst(veloc_mir::Float::from_f64_bits(0));
                ins.insert(data, &[Type::F32]);
            }
            "relation" => {
                let result = ins.sized(i);
                assert_eq!(ins.value_type(result), Type::I8);
            }
            "fixed" => {
                let result = ins.icmp(IntCC::Eq, f, i);
                assert_eq!(ins.value_type(result), Type::BOOL);
            }
            "raw-results" | "raw-arity" => {
                let data = InstDraft::from_values(Opcode::IAdd, &[i, i]).unwrap();
                let types = if case == "raw-results" {
                    &[Type::F32][..]
                } else {
                    &[]
                };
                let inst = ins.insert(data, types);
                assert_eq!(
                    ins.builder().func().dfg().inst_results(inst).len(),
                    types.len()
                );
            }
            "call" => {
                let inst = ins.call(callee, &[i]);
                let dfg = ins.builder().func().dfg();
                let types = dfg
                    .inst_results(inst)
                    .iter()
                    .map(|&v| dfg.value_type(v))
                    .collect::<Vec<_>>();
                assert_eq!(types, [Type::I64, Type::I32]);
            }
            "indirect-call" => {
                let inst = ins.call_indirect(callee_sig, i, &[f]);
                assert_eq!(ins.builder().func().dfg().inst_results(inst).len(), 2);
            }
            "branch" | "table" => {
                let dest = ins.builder().create_block();
                if case == "branch" {
                    ins.br(i, dest, &[], dest, &[]);
                } else {
                    let call = ins.builder().make_block_call(dest, &[]);
                    ins.br_table(f, call, &[]);
                }
                ins.builder().switch_to_block(dest);
            }
            _ => unreachable!(),
        }
        ins.ret(&[]);
        drop(builder);
        let error = module.validate().unwrap_err().to_string();
        assert!(error.contains(expected), "{case}: {error}");
        // A Float's precision is supplied by its result annotation in text;
        // inconsistent typed properties are tested through raw construction.
        if !case.starts_with("raw-") && case != "float-type" {
            // The textual path constructs the same invalid IR, and only the
            // explicit validator rejects it there as well.
            let text = module.build().to_string();
            let parsed = veloc_mir::ModuleParser::new().parse(&text).unwrap();
            assert!(
                parsed
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains(expected),
                "{case}"
            );
        }
    }
}

#[test]
fn result_resolution_only_requires_construction_inputs() {
    use veloc_mir::{Block, ModuleData, SigId};
    let mut dfg = veloc_mir::dfg::DataFlowGraph::new();
    let module = ModuleData::default();
    let i = dfg.append_block_param(Block(0), Type::I32);
    let f = dfg.append_block_param(Block(0), Type::F32);
    let unknown = dfg.append_block_param(Block(0), Type::INVALID);
    let data = InstDraft::from_values(Opcode::First, &[i, f]).unwrap();
    assert_eq!(
        data.result_types(&dfg, &module, &[]).unwrap().as_slice(),
        &[Type::I32]
    );
    let data = InstDraft::from_values(Opcode::First, &[unknown, i]).unwrap();
    assert!(data.result_types(&dfg, &module, &[]).is_err());
    let data = InstDraft::from_values(Opcode::Sized, &[unknown]).unwrap();
    assert_eq!(
        data.result_types(&dfg, &module, &[]).unwrap().as_slice(),
        &[Type::I8]
    );
    let output = InstDraft::empty(Opcode::Output);
    assert!(output.result_types(&dfg, &module, &[]).is_err());
    assert_eq!(
        output
            .result_types(&dfg, &module, &[Type::F32])
            .unwrap()
            .as_slice(),
        &[Type::F32]
    );
    let data = InstDraft::from_values(Opcode::Lane, &[i]).unwrap();
    assert!(data.result_types(&dfg, &module, &[]).is_err());
    let data = InstDraft::call_indirect(i, &[], SigId(123));
    assert!(data.result_types(&dfg, &module, &[]).is_err());
}

#[test]
fn builders_preserve_logical_order_independently_of_storage_and_text() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(
        vec![Type::I32, Type::I32, Type::PTR],
        vec![Type::I32],
        CallConv::SystemV,
    );
    let id = module.declare_function("builders".into(), sig, Linkage::Local);
    let mut builder = module.builder(id);
    builder.init_entry_block();
    let a = builder.func_param(0);
    let b = builder.func_param(1);
    let ptr = builder.func_param(2);
    let difference = builder.ins().difference(a, b);
    let reverse = builder.ins().reverse_text(a, b);
    let selected = builder.ins().select_type(a, b, Type::I64);
    let (first, wide) = builder.ins().wide_pair(a, b);
    let triple = builder.ins().triple(a, b, difference);
    let output = builder.ins().output(Type::I32);
    let offset = builder.ins().offset(ptr, 7);
    let (last, first_arg) = builder.ins().many(a, b, a, b, ptr);
    let (address, integer, flag) = builder.ins().three_results(a, ptr);
    builder.ins().nop();
    builder.ins().ret(&[triple]);
    let dfg = builder.func().dfg();
    let inst = |value| dfg.inst(dfg.value_inst(value).unwrap());
    assert!(matches!(inst(difference), InstructionView::Pair { inputs, .. } if *inputs == [a, b]));
    assert!(
        matches!(inst(reverse), InstructionView::FieldPair { right, left, .. } if (right, left) == (a, b))
    );
    assert!(
        matches!(inst(triple), InstructionView::Triple { args, .. } if *args == [a, b, difference])
    );
    assert!(
        matches!(inst(offset), InstructionView::Immediate { value, displacement: 7, .. } if value == ptr)
    );
    assert_eq!(dfg.value_type(selected), Type::I64);
    assert_eq!(dfg.value_type(first), Type::I32);
    assert_eq!(dfg.value_type(wide), Type::I64);
    assert_eq!(dfg.value_type(output), Type::I32);
    assert_eq!(dfg.value_type(last), Type::PTR);
    assert_eq!(dfg.value_type(first_arg), Type::I32);
    assert_eq!(dfg.value_type(address), Type::PTR);
    assert_eq!(dfg.value_type(integer), Type::I32);
    assert_eq!(dfg.value_type(flag), Type::BOOL);
    let three_results = dfg.value_inst(address).unwrap();
    assert_eq!(dfg.inst_results(three_results), [address, integer, flag]);
    assert_eq!(dfg.value_inst(integer), Some(three_results));
    assert_eq!(dfg.value_inst(flag), Some(three_results));
    drop(builder);
    module.validate().unwrap();
    let module = module.build();
    let text = module.to_string();
    assert!(text.contains("reverse-text v1, v0"));
    assert!(!text.contains("amount="));
    veloc_mir::ModuleParser::new()
        .parse(&text)
        .unwrap()
        .validate()
        .unwrap();
    // Inferred text cannot construct mismatched result types, but callers of
    // the in-memory IR can. Check the generated validator independently too.
    let mut malformed = (*module).clone();
    malformed.functions[id]
        .edit()
        .set_value_type(last, Type::I32);
    malformed.functions[id]
        .edit()
        .set_value_type(first_arg, Type::PTR);
    assert!(
        malformed
            .validate()
            .unwrap_err()
            .to_string()
            .contains("many")
    );
}

#[test]
fn generated_predicates_and_inline_sets_observe_actual_types() {
    const { assert!(Type::I32.is_wide()) };
    for raw in 0..=u16::MAX {
        let Some(ty) = Type::from_raw(raw) else {
            continue;
        };
        assert_eq!(ty.is_wide(), ty == Type::I32 || ty == Type::I64);
        assert_eq!(ty.is_chosen(), ty == Type::I32X4 || ty == Type::SV4);
        let named = Opcode::Named.validate_types(&[ty, ty], &[ty]);
        let inline = Opcode::Inline.validate_types(&[ty, ty], &[ty]);
        assert_eq!(named.is_ok(), ty == Type::I32 || ty == Type::I64);
        assert_eq!(named, inline);
    }
    assert!(!Type::INVALID.is_wide());
    assert!(!Type::INVALID.is_chosen());
    assert!(
        Opcode::Named
            .validate_types(&[Type::I32, Type::I64], &[Type::I32])
            .is_err()
    );
}

#[test]
fn generated_comparison_transforms_follow_outcomes() {
    use veloc_mir::{FloatOrderCC as F, OrderCC as I};
    assert_eq!(I::Before.swap(), I::After);
    assert_eq!(I::NotBefore.swap(), I::NotAfter);
    assert_eq!(I::Before.complement(), I::NotBefore);
    assert_eq!(I::After.complement(), I::NotAfter);
    assert_eq!(I::from_mnemonic("notbefore"), Some(I::NotBefore));
    assert_eq!(F::Before.complement(), Some(F::NotBefore));
    assert_eq!(F::NotAfter.complement(), Some(F::After));
}

#[test]
fn generated_lowering_only_accepts_direct_nontrapping_primitives() {
    assert_eq!(
        direct_lowering(Opcode::Direct),
        Some(veloc_lir::GenericOpcode::G_SUB)
    );
    for opcode in [
        Opcode::Reversed,
        Opcode::Composed,
        Opcode::Trapping,
        Opcode::Multiple,
    ] {
        assert_eq!(direct_lowering(opcode), None);
    }
}

#[test]
fn generated_evaluators_execute_compositions_properties_and_traps() {
    for (opcode, args, results, properties, expected) in [
        (
            Opcode::Direct,
            vec![ScalarConst::from(7i32), ScalarConst::from(3i32)],
            vec![Type::I32],
            vec![],
            Some(vec![ScalarConst::from(4i32)]),
        ),
        (
            Opcode::Reversed,
            vec![ScalarConst::from(7i32), ScalarConst::from(3i32)],
            vec![Type::I32],
            vec![],
            Some(vec![ScalarConst::from(-4i32)]),
        ),
        (
            Opcode::Composed,
            vec![ScalarConst::from(7i32), ScalarConst::from(3i32)],
            vec![Type::I32],
            vec![],
            Some(vec![ScalarConst::from(9i32)]),
        ),
        (
            Opcode::Trapping,
            vec![ScalarConst::from(7i32), ScalarConst::from(0i32)],
            vec![Type::I32],
            vec![],
            None,
        ),
        (
            Opcode::Trapping,
            vec![ScalarConst::from(7i32), ScalarConst::from(3i32)],
            vec![Type::I32],
            vec![],
            Some(vec![ScalarConst::from(2i32)]),
        ),
        (
            Opcode::Multiple,
            vec![ScalarConst::from(7i32), ScalarConst::from(7i32)],
            vec![Type::I32, Type::BOOL],
            vec![],
            Some(vec![ScalarConst::from(14i32), ScalarConst::from(true)]),
        ),
        (
            Opcode::CompareValue,
            vec![ScalarConst::from(7i32), ScalarConst::from(3i32)],
            vec![Type::BOOL],
            vec![IntCC::GtS],
            Some(vec![ScalarConst::from(true)]),
        ),
        (
            Opcode::ExtendS,
            vec![ScalarConst::from(-1i8)],
            vec![Type::I64],
            vec![],
            Some(vec![ScalarConst::from(-1i64)]),
        ),
        (
            Opcode::ExtendS,
            vec![ScalarConst::from(-1i64)],
            vec![Type::I8],
            vec![],
            None,
        ),
    ] {
        assert_eq!(
            evaluator::evaluate(opcode, &args, &results, &properties),
            expected,
            "{opcode:?}"
        );
    }
    assert!(!evaluator::can_fold(Opcode::VectorOnly));
    assert!(!evaluator::can_fold(Opcode::Difference));
    let compare = InstDraft::compare(Opcode::CompareValue, IntCC::GtS, [Value(0), Value(1)]);
    assert_eq!(
        evaluator::properties(&compare.as_view()).as_slice(),
        &[IntCC::GtS]
    );
    assert!(!Opcode::Composed.spec().is_commutative());
    assert!(!Opcode::Composed.spec().is_associative());
    assert!(Opcode::Trapping.spec().may_trap());
    let spec = offline::SPECS
        .iter()
        .find(|spec| spec.opcode == Opcode::Trapping)
        .unwrap();
    let sort = veloc_semantics::Sort::bv(32).unwrap();
    let function = spec
        .program
        .instantiate(&[sort, sort], &[sort], &[])
        .unwrap();
    assert_eq!(
        function
            .execute(&[veloc_semantics::Value::Bv(7), veloc_semantics::Value::Bv(0)])
            .unwrap(),
        veloc_semantics::Outcome::Trap(veloc_semantics::Trap::DivisionByZero)
    );
}

#[test]
fn drafts_share_storage_shape_and_edit_repeated_successors_independently() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::{Block, Successor};
    let target = Block(1);
    let mut dfg = DataFlowGraph::new();
    let inst = dfg.create_inst(InstDraft::br_table(
        Value(0),
        [
            Successor {
                block: target,
                args: &[Value(1)],
            },
            Successor {
                block: Block(2),
                args: &[Value(2), Value(3)],
            },
            Successor {
                block: target,
                args: &[],
            },
        ],
    ));
    let original = dfg.operands(inst).to_vec();
    let mut draft = dfg.draft(inst);
    let untouched = draft.clone();
    draft.edit_successors(|edge| {
        if edge.block() == target {
            edge.set_arg(2, Value(4));
        }
    });
    draft.set_operand(0, Value(5));
    assert_eq!(dfg.operands(inst), original);
    assert_eq!(untouched.operands(), original);
    assert_eq!(
        draft.operands(),
        [
            Value(5),
            Value(1),
            Value(4),
            Value(4),
            Value(2),
            Value(3),
            Value(4),
            Value(4),
            Value(4),
        ]
    );
    let InstructionView::BrTable { index, table } = draft.as_view() else {
        unreachable!()
    };
    assert_eq!(index, Value(5));
    let calls: Vec<_> = table.iter().collect();
    assert_eq!(
        calls.iter().map(|c| c.block).collect::<Vec<_>>(),
        [target, Block(2), target]
    );
    assert_eq!(calls[0].args, &[Value(1), Value(4), Value(4)]);
    assert_eq!(calls[1].args, &[Value(2), Value(3)]);
    assert_eq!(table.split_last().unwrap().0.args, &[Value(4); 3]);
    dfg.replace_inst(inst, draft);
    assert_eq!(
        dfg.operands(inst),
        [
            Value(5),
            Value(1),
            Value(4),
            Value(4),
            Value(2),
            Value(3),
            Value(4),
            Value(4),
            Value(4),
        ]
    );
    dfg.check_uses().unwrap();
    assert!(dfg.use_empty(Value(0)));
    assert_eq!(dfg.uses(Value(4)).count(), 5);
}

#[test]
fn draft_successor_growth_preserves_record_inputs_and_following_fields() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::{TestOperands, VectorExtData};
    use veloc_mir::{Block, Successor};
    let mut dfg = DataFlowGraph::new();
    for optional in [None, Some(Value(1))] {
        let mut draft = InstDraft::routed(
            Opcode::Routed,
            Value(3),
            TestOperands {
                token: Value(0),
                optional,
                tag: 19,
            },
            Successor {
                block: Block(1),
                args: &[Value(2)],
            },
            VectorExtData {
                mask: Value(5),
                evl: optional,
            },
        );
        draft.edit_successors(|edge| edge.set_arg(3, Value(4)));
        let InstructionView::Routed {
            group,
            dest,
            operands,
            tail,
            ..
        } = draft.as_view()
        else {
            unreachable!()
        };
        assert_eq!(
            (group.token, group.optional, group.tag),
            (Value(0), optional, 19)
        );
        assert_eq!(dest.args, &[Value(2), Value(4), Value(4), Value(4)]);
        assert_eq!(operands, Value(3));
        assert_eq!((tail.mask, tail.evl), (Value(5), optional));
        let expected = draft.operands().to_vec();
        let inst = dfg.create_inst(draft);
        assert_eq!(dfg.operands(inst), expected);
        assert_eq!(dfg.draft(inst).operands(), expected);
        dfg.check_uses().unwrap();
    }
}
