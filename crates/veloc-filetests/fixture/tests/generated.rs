//! Compile and execute generated APIs against the real MIR implementation.
//! File tests cover text/diagnostics; these cover APIs text cannot exercise.
use veloc_types::TypeInfo;
extern crate alloc;
extern crate veloc_test_mir as veloc_mir;

use veloc_mir::constant::ScalarConst;
use veloc_mir::{
    Arguments, BlockCall, CallConv, InstView, IntCC, Linkage, ModuleBuilder, Opcode, Type, Value,
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

#[test]
fn shared_expressions_execute_in_queries_and_explicit_validation() {
    use veloc_mir::{dfg::DataFlowGraph, inst::SizeInfo};
    let mut dfg = DataFlowGraph::new();
    for (size, expected, valid) in [
        (
            8,
            Some(SizeInfo {
                next: 9,
                aligned: true,
            }),
            true,
        ),
        (
            0,
            Some(SizeInfo {
                next: 1,
                aligned: false,
            }),
            false,
        ),
        (
            3,
            Some(SizeInfo {
                next: 4,
                aligned: false,
            }),
            false,
        ),
        (u32::MAX, None, false),
    ] {
        // Construction and parsing intentionally accept invalid contracts.
        let inst = dfg.writer().checked_size(size);
        assert_eq!(inst.size_info(&dfg), expected);
        assert_eq!(
            inst.original_size(&dfg),
            Some(SizeInfo {
                next: size,
                aligned: false
            })
        );
        let text = format!(
            "local function test() -> void\nblock0():\n  checked-size size={size}\n  return\n"
        );
        let module = veloc_mir::ModuleParser::new().parse(&text).unwrap();
        assert_eq!(module.validate().is_ok(), valid);
    }
}

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
fn instruction_queries_use_their_own_results() {
    use veloc_mir::{Block, MemFlags, dfg::DataFlowGraph};
    let mut dfg = DataFlowGraph::new();
    dfg.create_block();
    let ptr = dfg.append_block_param(Block(0), Type::PTR);
    let flags = MemFlags::empty();
    let first = dfg.create_inst_with_results(|w| w.load(ptr, 4, flags), &[Type::I32]);
    let second = dfg.create_inst_with_results(|w| w.load(ptr, 8, flags), &[Type::I64]);
    for (inst, ty, offset) in [(first, Type::I32, 4), (second, Type::I64, 8)] {
        let access = inst.memory_access(&dfg).unwrap();
        assert_eq!((access.ptr, access.ty, access.offset), (ptr, ty, offset));
        assert!(access.stored.is_none());
    }
    let value = dfg.first_result(second).unwrap();
    let store = dfg.writer().store(ptr, value, 12, flags);
    let access = store.memory_access(&dfg).unwrap();
    assert_eq!(access.ty, Type::I64);
    assert_eq!(access.stored, Some(value));
    let unrelated = dfg.writer().checked_size(8);
    assert!(unrelated.memory_access(&dfg).is_none());
}

#[test]
fn host_queries_preserve_optional_results_through_helpers() {
    use veloc_mir::{Block, CallableKind, SigId, dfg::DataFlowGraph};
    let mut dfg = DataFlowGraph::new();
    dfg.create_block();
    for (ty, expected) in [
        (
            Type::callable(SigId(7), CallableKind::Local),
            Some(SigId(7)),
        ),
        (Type::I32, None),
    ] {
        let value = dfg.append_block_param(Block(0), ty);
        let inst = dfg.writer().unary(Opcode::Rebind, value);
        let info = inst.callable_info(&dfg).unwrap();
        assert_eq!(info.signature, expected);
    }
}

#[test]
fn definition_owned_records_flatten_operands_in_field_order() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::TestOperands;
    let mut dfg = DataFlowGraph::new();
    for optional in [None, Some(Value(1))] {
        let inst = dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| {
            writer.grouped(
                Value(0),
                TestOperands {
                    token: Value(1),
                    optional,
                    tag: 17,
                },
                Value(2),
            )
        });
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
        let InstView::Grouped {
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
    }
}

#[test]
fn variadic_ranges_grow_recycle_and_remain_independent_after_clone() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let left = dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| writer.nop());
    let right = dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| {
        writer.binary(Opcode::IAdd, [Value(7), Value(7)])
    });
    for size in [0, 1, 2, 3, 4, 5, 17, 65, 3, 0, 65, 17, 5, 1] {
        let args: Arguments = (0..size).map(|i| Value(i % 7)).collect();
        dfg.replace_inst(left, |writer: veloc_mir::InstWriter<'_>| writer.ret(&args));
        assert_eq!(dfg.operands(left), args.as_slice());
        assert_eq!(dfg.operands(right), &[Value(7), Value(7)]);
        let InstView::Return { values } = dfg.inst(left) else {
            unreachable!()
        };
        assert_eq!(values.as_ptr(), dfg.operands(left).as_ptr());
    }
    let mut copy = dfg.clone();
    copy.replace_all_uses(Value(7), Value(8));
    assert_eq!(dfg.operands(right), &[Value(7), Value(7)]);
    assert_eq!(copy.operands(right), &[Value(8), Value(8)]);
}

#[test]
fn function_edits_keep_layout_and_successor_edges_in_sync() {
    let module = veloc_mir::ModuleParser::new().parse("local function main() -> void\nblock0():\n  jump block1()\nblock1():\n  return\nblock2():\n  return\n").unwrap();
    let mut data = (*module).clone();
    let func = data
        .bodies
        .iter_mut()
        .find_map(|(_, body)| body.as_deref_mut())
        .unwrap();
    let entry = func.entry_block();
    let old = func.layout().block_order().nth(1).unwrap();
    let new = func.layout().block_order().nth(2).unwrap();
    let jump = func.layout().first_inst(entry).unwrap();
    let dest = BlockCall::new(new, &[]);
    func.edit()
        .replace_inst(jump, |writer: veloc_mir::InstWriter<'_>| {
            writer.jump(dest.as_view())
        });
    assert!(func.cfg().blocks()[old].preds.is_empty());
    assert_eq!(func.cfg().blocks()[new].preds, [entry]);
    assert_eq!(func.cfg().blocks()[entry].succs, [new]);
    func.edit().erase_inst(jump);
    assert!(func.layout().inst_block(jump).is_none());
    assert!(func.cfg().blocks()[new].preds.is_empty());
    assert!(func.cfg().blocks()[entry].succs.is_empty());
    let replacement = func.edit().append_inst(
        entry,
        |writer: veloc_mir::InstWriter<'_>| writer.jump(dest.as_view()),
        &[],
    );
    assert_eq!(func.layout().inst_block(replacement), Some(entry));

    // Exercise both ends and mixed-direction iteration after in-place edits.
    let first = func.edit().prepend_inst(entry, |w| w.nop(), &[]);
    let middle = func.edit().insert_after(first, |w| w.nop(), &[]);
    let last = func.edit().insert_before(replacement, |w| w.nop(), &[]);
    assert_eq!(
        func.layout().block_insts(entry).collect::<Vec<_>>(),
        [first, middle, last, replacement]
    );
    {
        let mut order = func.layout().block_insts(entry);
        assert_eq!(order.next(), Some(first));
        assert_eq!(order.next_back(), Some(replacement));
        assert_eq!(order.next_back(), Some(last));
        assert_eq!(order.next(), Some(middle));
        assert_eq!(order.next_back(), None);
        assert_eq!(order.next(), None);
    }
    func.edit().move_before(last, first);
    func.edit().move_before(last, last);
    assert_eq!(func.layout().prev_inst(first), Some(last));
    assert_eq!(func.layout().next_inst(last), Some(first));

    let ret = func.layout().last_inst(old).unwrap();
    func.edit().move_before(middle, ret);
    assert_eq!(func.layout().inst_block(middle), Some(old));
    func.edit().erase_insts(&[last, first, middle]);
    assert_eq!(
        func.layout().block_insts(entry).collect::<Vec<_>>(),
        [replacement]
    );

    // Temporary edits may invalidate terminator placement; CFG still follows
    // the actual tail, and explicit validation runs once the edit is complete.
    let tail = func.edit().insert_after(replacement, |w| w.nop(), &[]);
    assert!(func.cfg().blocks()[entry].succs.is_empty());
    func.edit().erase_inst(tail);
    assert_eq!(func.cfg().blocks()[entry].succs, [new]);

    // Move a terminator away and back, updating both sides of cached CFG edges.
    func.edit().move_to_end(replacement, old);
    assert!(func.cfg().blocks()[entry].succs.is_empty());
    assert_eq!(func.cfg().blocks()[old].succs, [new]);
    assert_eq!(func.cfg().blocks()[new].preds, [old]);
    func.edit().move_to_end(replacement, entry);
    assert_eq!(func.cfg().blocks()[new].preds, [entry]);
    assert!(func.cfg().blocks()[old].succs.is_empty());

    func.edit().move_block_before(new, entry);
    assert_eq!(
        func.layout().block_order().collect::<Vec<_>>(),
        [new, entry, old]
    );
    func.edit().move_block_before(new, old);
    func.edit().move_block_before(old, new);
    func.edit().move_block_before(entry, entry);
    assert_eq!(
        func.layout().block_order().rev().collect::<Vec<_>>(),
        [new, old, entry]
    );
    assert_eq!(func.entry_block(), entry);
    data.validate().unwrap();
}

#[test]
fn borrowed_uses_distinguish_operands_and_edits_update_the_single_storage() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let inst = dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| {
        writer.binary(Opcode::IAdd, [Value(0), Value(0)])
    });
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
    let before = dfg.clone();
    dfg.replace_inst(inst, |writer: veloc_mir::InstWriter<'_>| {
        writer.unary(Opcode::INeg, Value(2))
    });
    let invalid = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dfg.set_operand(inst, 1, Value(3))
    }));
    assert!(invalid.is_err());
    assert_eq!(before.uses(Value(1)).count(), 2);
    assert!(dfg.use_empty(Value(1)));
}

#[test]
fn unified_slots_distinguish_repeated_successors_and_vector_operands() {
    use veloc_mir::Block;
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::VectorExtData;
    let mut dfg = DataFlowGraph::new();
    let call = BlockCall::new(Block(0), &[Value(0)]);
    let table = [call.clone(), call.clone()];
    let data = |writer: veloc_mir::InstWriter<'_>| {
        writer.br_table(Value(0), table.iter().map(BlockCall::as_view))
    };
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
    let data =
        |writer: veloc_mir::InstWriter<'_>| writer.vector_op_with_ext(Opcode::IAdd, &args, ext);
    let first = dfg.create_inst(data.clone());
    let other = dfg.create_inst(data);
    dfg.set_operand(first, 2, Value(2));
    assert_eq!(dfg.operands(first)[3], Value(0));
    assert_eq!(dfg.operands(other)[2], Value(0));
    assert_eq!(ext.mask, Value(0));
    dfg.replace_all_uses(Value(0), Value(3));
}

#[test]
fn closed_dead_cycles_are_erased_together() {
    use veloc_mir::dfg::DataFlowGraph;
    let mut dfg = DataFlowGraph::new();
    let a = dfg.create_inst_with_results(
        |writer: veloc_mir::InstWriter<'_>| writer.unary(Opcode::INeg, Value(1)),
        &[Type::I32],
    );
    let b = dfg.create_inst_with_results(
        |writer: veloc_mir::InstWriter<'_>| writer.unary(Opcode::INeg, Value(0)),
        &[Type::I32],
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dfg.remove_inst(a))).is_err());
    dfg.remove_insts(&[a, b]);
    assert!(dfg.use_empty(Value(0)) && dfg.use_empty(Value(1)));
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
fn scalar_view_preserves_type_encoding() {
    use veloc_mir::{CallableKind, ScalarType, SigId};

    fn width(ty: ScalarType) -> Option<u32> {
        ty.element().element_bits()
    }
    assert_eq!(size_of::<ScalarType>(), size_of::<Type>());
    assert_eq!(size_of::<Type>(), 8);
    const SCALAR: ScalarType = Type::I32.as_scalar().unwrap();
    const VECTOR: Type = SCALAR.vector(4, false).unwrap().as_type();
    assert_eq!(SCALAR, ScalarType::I32);
    assert_eq!(VECTOR, veloc_mir::Type::I32X4);
    for code in 0..=u8::MAX {
        let scalar = ScalarType::from_code(code);
        assert_eq!(
            Type::from_scalar_code(code),
            scalar.map(ScalarType::as_type)
        );
        if let Some(scalar) = scalar {
            assert_eq!(scalar.code(), code);
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
fn rust_flag_sets_are_used_by_generated_metadata() {
    use veloc_mir::inst::{MemoryEffect, MemoryEffects, OpTraits};
    assert_eq!(size_of::<OpTraits>(), 2);
    assert_eq!(size_of::<MemoryEffects>(), 1);
    const EFFECT: MemoryEffect = Opcode::EffectSet.spec().memory_effect();
    assert_eq!(
        EFFECT,
        MemoryEffect::Known(MemoryEffects::READ.union(MemoryEffects::WRITE))
    );
    assert_eq!(EFFECT.to_string(), "READ | WRITE");
    assert_eq!(OpTraits::MAY_TRAP.to_string(), "MAY_TRAP");
    assert_eq!(MemoryEffects::empty().to_string(), "");
    assert!(MemoryEffects::from_bits(0x80).is_none());
    let unknown = MemoryEffects::from_bits_retain(0x80);
    assert_eq!(unknown.to_string(), "0x80");
    assert_eq!(
        unknown.union(MemoryEffects::READ).to_string(),
        "READ | 0x80"
    );
    assert!(MemoryEffect::Known(unknown).conflicts_with(MemoryEffect::Known(MemoryEffects::READ)));
    assert_eq!(
        Opcode::Nop.spec().memory_effect(),
        MemoryEffect::Known(MemoryEffects::empty())
    );
    assert_eq!(
        Opcode::Load.spec().memory_effect(),
        MemoryEffect::Known(MemoryEffects::READ)
    );
    assert_eq!(Opcode::Call.spec().memory_effect(), MemoryEffect::Unknown);
    assert_ne!(
        MemoryEffect::Known(MemoryEffects::all()),
        MemoryEffect::Unknown
    );
    assert_eq!(
        OpTraits::TERMINATOR
            .union(OpTraits::COMMUTATIVE)
            .to_string(),
        "TERMINATOR | COMMUTATIVE"
    );
}

#[test]
fn declaration_records_enums_and_flags_construct_explicit_values() {
    use veloc_mir::inst::{MemoryEffects, TestMetadata, TestPolicy, TestSettings};
    let meta = TestMetadata {
        settings: TestSettings {
            enabled: true,
            policy: TestPolicy::Prefer(MemoryEffects::READ.union(MemoryEffects::WRITE)),
        },
        fallback: Some(TestPolicy::Automatic),
    };
    assert!(meta.settings.enabled);
    assert_eq!(
        meta.settings.policy,
        TestPolicy::Prefer(MemoryEffects::READ.union(MemoryEffects::WRITE))
    );
    assert_eq!(meta.fallback, Some(TestPolicy::Automatic));
    assert!(core::ptr::eq(
        Opcode::IAdd.meta(),
        &Opcode::IAdd.spec().meta
    ));
    assert!(
        Opcode::IAdd
            .meta()
            .traits
            .contains(veloc_mir::inst::OpTraits::COMMUTATIVE)
    );
    assert_eq!(
        Opcode::IAdd.meta().memory,
        veloc_mir::inst::MemoryEffect::Known(veloc_mir::inst::MemoryEffects::empty())
    );
}

#[test]
fn construction_does_not_validate_type_contracts() {
    for (case, expected) in [
        ("binding", "Pattern { results: false, index: 1"),
        ("typeset", "Pattern { results: false, index: 0"),
        ("explicit", "Pattern { results: true, index: 0"),
        ("float-type", "result 0 must have the type of `value`"),
        ("relation", "result must have more bits"),
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
        let mut builder = module.define(id);
        let mut ins = builder.ins();
        let i = ins.i32const(1);
        let f = ins.f32const(1.0);
        match case {
            "binding" => {
                let result = ins.first(i, f);
                assert_eq!(ins.value_type(result), Type::I32);
            }
            "typeset" => {
                let result = ins.float_only(i, i);
                assert_eq!(ins.value_type(result), Type::I32);
            }
            "explicit" => {
                let result = ins.output(Type::F32);
                assert_eq!(ins.value_type(result), Type::F32);
            }
            "float-type" => {
                let data = |writer: veloc_mir::InstWriter<'_>| {
                    writer.fconst(veloc_mir::Float::from_f64_bits(0))
                };
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
                let data = |writer: veloc_mir::InstWriter<'_>| {
                    writer.from_values(Opcode::IAdd, &[i, i]).unwrap()
                };
                let types = if case == "raw-results" {
                    &[Type::F32][..]
                } else {
                    &[]
                };
                let inst = ins.insert(data, types);
                assert_eq!(ins.dfg().inst_results(inst).len(), types.len());
            }
            "call" => {
                let inst = ins.call(callee, &[i]);
                let dfg = ins.dfg();
                let types = dfg
                    .inst_results(inst)
                    .iter()
                    .map(|&v| dfg.value_type(v))
                    .collect::<Vec<_>>();
                assert_eq!(types, [Type::I64, Type::I32]);
            }
            "indirect-call" => {
                let inst = ins.call_indirect(callee_sig, i, &[f]);
                assert_eq!(ins.dfg().inst_results(inst).len(), 2);
            }
            "branch" | "table" => {
                drop(ins);
                let dest = builder.create_block();
                ins = builder.ins();
                if case == "branch" {
                    ins.br(i, dest, &[], dest, &[]);
                } else {
                    let call = veloc_mir::BlockCall::new(dest, &[]);
                    ins.br_table(f, call, &[]);
                }
                drop(ins);
                builder.switch_to_block(dest);
                ins = builder.ins();
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
fn builders_preserve_logical_order_independently_of_storage_and_text() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(
        vec![Type::I32, Type::I32, Type::PTR],
        vec![Type::I32],
        CallConv::SystemV,
    );
    let id = module.declare_function("builders".into(), sig, Linkage::Local);
    let mut builder = module.define(id);
    let a = builder.func().params()[0];
    let b = builder.func().params()[1];
    let ptr = builder.func().params()[2];
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
    assert!(matches!(inst(difference), InstView::Pair { inputs, .. } if *inputs == [a, b]));
    assert!(
        matches!(inst(reverse), InstView::FieldPair { right, left, .. } if (right, left) == (a, b))
    );
    assert!(matches!(inst(triple), InstView::Triple { args, .. } if *args == [a, b, difference]));
    assert!(
        matches!(inst(offset), InstView::Immediate { value, displacement: 7, .. } if value == ptr)
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
    assert!(text.contains("amount=7"));
    veloc_mir::ModuleParser::new()
        .parse(&text)
        .unwrap()
        .validate()
        .unwrap();
    // Inferred text cannot construct mismatched result types, but callers of
    // the in-memory IR can. Check the generated validator independently too.
    let mut malformed = (*module).clone();
    malformed.bodies[id]
        .as_deref_mut()
        .unwrap()
        .edit()
        .set_value_type(last, Type::I32);
    malformed.bodies[id]
        .as_deref_mut()
        .unwrap()
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
fn named_and_inline_sets_observe_actual_types() {
    for raw in 0..=u16::MAX {
        let Some(ty) = Type::from_raw(raw) else {
            continue;
        };
        let named = Opcode::Named.validate_types(&[ty, ty], &[ty]);
        let inline = Opcode::Inline.validate_types(&[ty, ty], &[ty]);
        assert_eq!(named.is_ok(), ty == Type::I32 || ty == Type::I64);
        assert_eq!(named, inline);
    }
    assert!(
        Opcode::Named
            .validate_types(&[Type::I32, Type::I64], &[Type::I32])
            .is_err()
    );
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
    // The generated evaluator and offline graph consume the same Rust-owned
    // condition codes, without generator-side predicate tables.
    let compare = offline::SPECS
        .iter()
        .find(|spec| spec.opcode == Opcode::CompareValue)
        .unwrap();
    let sort = veloc_semantics::Sort::bv(32).unwrap();
    for cc in [
        IntCC::Eq,
        IntCC::Ne,
        IntCC::LtS,
        IntCC::LtU,
        IntCC::GtS,
        IntCC::GtU,
        IntCC::LeS,
        IntCC::LeU,
        IntCC::GeS,
        IntCC::GeU,
    ] {
        let function = compare
            .program
            .instantiate(
                &[sort, sort],
                &[veloc_semantics::Sort::Bool],
                &[offline::predicate(cc)],
            )
            .unwrap();
        for lhs in [i32::MIN, -1, 0, 1, i32::MAX] {
            for rhs in [i32::MIN, -1, 0, 1, i32::MAX] {
                let expected = cc.test(32, lhs as u32 as u128, rhs as u32 as u128);
                assert_eq!(
                    evaluator::evaluate(
                        Opcode::CompareValue,
                        &[ScalarConst::from(lhs), ScalarConst::from(rhs)],
                        &[Type::BOOL],
                        &[cc]
                    ),
                    Some(vec![ScalarConst::from(expected)])
                );
                assert_eq!(
                    function
                        .execute(&[
                            veloc_semantics::Value::Bv(lhs as u32 as u128),
                            veloc_semantics::Value::Bv(rhs as u32 as u128)
                        ])
                        .unwrap(),
                    veloc_semantics::Outcome::Values(vec![veloc_semantics::Value::Bool(expected)])
                );
            }
        }
    }
    assert!(!evaluator::can_fold(Opcode::VectorOnly));
    assert!(!evaluator::can_fold(Opcode::Difference));
    let mut dfg = veloc_mir::dfg::DataFlowGraph::new();
    let compare = dfg.writer().compare(IntCC::GtS, [Value(0), Value(1)]);
    assert_eq!(
        evaluator::properties(&dfg.inst(compare)).as_slice(),
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
fn pooled_copies_edit_repeated_successors_independently() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::{Block, Successor};
    let target = Block(1);
    let mut dfg = DataFlowGraph::new();
    let inst = dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| {
        writer.routed_table(
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
        )
    });
    let original = dfg.operands(inst).to_vec();
    let untouched = dfg.create_inst(|writer| writer.copy(inst));
    dfg.edit_successors(inst, |edge| {
        if edge.block() == target {
            edge.set_arg(2, Value(4));
        }
    });
    dfg.set_operand(inst, 0, Value(5));
    assert_eq!(dfg.operands(untouched), original);
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
    let InstView::RoutedTable {
        index,
        targets: table,
    } = dfg.inst(inst)
    else {
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
    dfg.remove_inst(untouched);
    assert!(dfg.use_empty(Value(0)));
    assert_eq!(dfg.uses(Value(4)).count(), 5);
}

#[test]
fn successor_growth_preserves_record_inputs_and_following_fields() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::{TestOperands, VectorExtData};
    use veloc_mir::{Block, Successor};
    let mut dfg = DataFlowGraph::new();
    for optional in [None, Some(Value(1))] {
        let build = |writer: veloc_mir::InstWriter<'_>| {
            writer.routed(
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
            )
        };
        let inst = dfg.create_inst(build);
        dfg.edit_successors(inst, |edge| edge.set_arg(3, Value(4)));
        let InstView::Routed {
            group,
            dest,
            operands,
            tail,
            ..
        } = dfg.inst(inst)
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
        let expected = dfg.operands(inst).to_vec();
        let copy = dfg.writer().copy(inst);
        assert_eq!(dfg.operands(copy), expected);
    }
}

#[test]
fn a_record_can_be_both_instruction_storage_and_plain_data() {
    use veloc_mir::dfg::DataFlowGraph;
    use veloc_mir::inst::LiteralPayload;

    let mut dfg = DataFlowGraph::new();
    let payload = LiteralPayload { bits: 42 };
    let direct =
        dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| writer.literal_payload(payload.bits));
    let nested =
        dfg.create_inst(|writer: veloc_mir::InstWriter<'_>| writer.wrapped_payload(payload));
    assert_eq!(dfg.inst(direct).opcode(), Opcode::Payload);
    assert_eq!(dfg.inst(nested).opcode(), Opcode::Wrapped);
    assert!(matches!(dfg.inst(nested),
        InstView::WrappedPayload { inner } if inner == payload));
    assert_eq!(
        core::mem::size_of::<LiteralPayload>(),
        core::mem::size_of::<u32>()
    );
}

#[test]
fn rust_type_bindings_preserve_paths_in_records_enums_and_host_queries() {
    use veloc_mir::{
        dfg::DataFlowGraph,
        inst::{StampRecord, StampResult},
        tokens::Stamp,
    };
    let mut dfg = DataFlowGraph::new();
    let inst = dfg.writer().stamp_input(17);
    let info = inst
        .stamp_info(&dfg, &veloc_mir::tokens::Tokens(&0))
        .unwrap();
    assert_eq!(info.stamp, Stamp(17));
    assert_eq!(info.doubled, 34);
    let record = StampRecord { stamp: info.stamp };
    assert_eq!(
        StampResult::Present(record.stamp),
        StampResult::Present(Stamp(17))
    );
    let module = veloc_mir::ModuleParser::new()
        .parse("local function test() -> void\nblock0():\n  read-stamp 17\n  return\n")
        .unwrap();
    module.validate().unwrap();
    let printed = module.to_string();
    veloc_mir::ModuleParser::new()
        .parse(&printed)
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn type_constraints_drive_validation_and_generated_evaluation() {
    for (from, to, valid) in [
        (Type::I8, Type::I16, true),
        (Type::I8, veloc_mir::Type::I16X8, false),
        (veloc_mir::Type::I8X16, Type::I16, false),
        (veloc_mir::Type::I8X16, veloc_mir::Type::I16X8, false),
        (
            veloc_mir::Type::I32X4,
            Type::I64
                .as_scalar()
                .unwrap()
                .vector(4, true)
                .unwrap()
                .as_type(),
            false,
        ),
        (
            Type::I32
                .as_scalar()
                .unwrap()
                .vector(4, true)
                .unwrap()
                .as_type(),
            Type::I64
                .as_scalar()
                .unwrap()
                .vector(4, true)
                .unwrap()
                .as_type(),
            true,
        ),
        (Type::I8, Type::I32, false),
        (Type::I64, Type::I32, false),
        (
            veloc_mir::Type::I32X4,
            Type::I64
                .as_scalar()
                .unwrap()
                .vector(4, false)
                .unwrap()
                .as_type(),
            true,
        ),
        (veloc_mir::Type::I64X2, veloc_mir::Type::I64X2, false),
    ] {
        assert_eq!(
            Opcode::DoubleWidth.validate_types(&[from], &[to]).is_ok(),
            valid
        );
    }
    // Construction remains independent of type/DFG validation.
    let draft = |writer: veloc_mir::InstWriter<'_>| writer.unary(Opcode::DoubleWidth, Value(999));
    let mut dfg = veloc_mir::dfg::DataFlowGraph::new();
    let inst = dfg.create_inst(draft);
    assert_eq!(dfg.inst(inst).opcode(), Opcode::DoubleWidth);
    assert_eq!(
        evaluator::evaluate(
            Opcode::DoubleWidth,
            &[ScalarConst::from(-1i8)],
            &[Type::I16],
            &[]
        ),
        Some(vec![ScalarConst::from(-1i16)])
    );
    assert_eq!(
        evaluator::evaluate(
            Opcode::DoubleWidth,
            &[ScalarConst::from(-1i8)],
            &[Type::I32],
            &[]
        ),
        None
    );
    assert!(
        Opcode::FourLane
            .validate_types(&[veloc_mir::Type::I32X4], &[veloc_mir::Type::I32X4])
            .is_ok()
    );
    assert!(
        Opcode::FourLane
            .validate_types(&[Type::I32], &[Type::I32])
            .is_err()
    );
    let scalable4 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, true)
        .unwrap()
        .as_type();
    assert!(
        Opcode::RuntimeScalable
            .validate_types(&[scalable4], &[scalable4])
            .is_ok()
    );
    for ty in [Type::I32, veloc_mir::Type::I32X4] {
        assert!(
            Opcode::RuntimeScalable
                .validate_types(&[ty], &[ty])
                .is_err()
        );
    }
    // Vector-admitted lanes must not leak into scalar constant evaluation.
    assert!(!evaluator::can_fold(Opcode::FourLane));
    assert!(
        Opcode::Reinterpret
            .validate_types(&[veloc_mir::Type::I32X4], &[veloc_mir::Type::I64X2])
            .is_ok()
    );
    let scalable = Type::I64
        .as_scalar()
        .unwrap()
        .vector(2, true)
        .unwrap()
        .as_type();
    assert!(
        Opcode::Reinterpret
            .validate_types(&[veloc_mir::Type::I32X4], &[scalable])
            .is_err()
    );
    assert!(
        Opcode::Reinterpret
            .validate_types(&[Type::I32], &[Type::I32])
            .is_err()
    );
}
