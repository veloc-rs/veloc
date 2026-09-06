//! Compile and execute generated APIs against the real MIR implementation.
//! File tests cover text/diagnostics; these cover APIs text cannot exercise.
extern crate alloc;
extern crate veloc_test_mir as veloc_mir;

use veloc_mir::constant::Constant;
use veloc_mir::{CallConv, InstructionData, IntCC, Linkage, ModuleBuilder, Opcode, Type, Value};

#[allow(dead_code)]
mod evaluator {
    use super::*;
    enum Replacement {
        Constants(Vec<Constant>),
        Value(Value),
    }
    include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));
}
mod offline {
    include!(concat!(env!("OUT_DIR"), "/semantics.rs"));
}
include!(concat!(env!("OUT_DIR"), "/lowering.rs"));

#[test]
fn generated_encodings_preserve_neighboring_fields_and_check_ranges() {
    use veloc_mir::opcode::{MemFlags, PackedRecord, WideRecord};
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
fn generated_flag_sets_preserve_bits_order_and_set_operations() {
    use veloc_mir::opcode::{EmptyFlags, MemoryRegions, OpTraits, TestFlags as F};
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
            "relation" => {
                let result = ins.sized(i);
                assert_eq!(ins.value_type(result), Type::I8);
            }
            "fixed" => {
                let result = ins.icmp(IntCC::Eq, f, i);
                assert_eq!(ins.value_type(result), Type::BOOL);
            }
            "raw-results" | "raw-arity" => {
                let data = InstructionData::from_values(Opcode::IAdd, &[i, i]).unwrap();
                let types = if case == "raw-results" {
                    &[Type::F32][..]
                } else {
                    &[]
                };
                let inst = ins.insert(data, types);
                assert_eq!(
                    ins.builder().func().dfg.inst_results(inst).len(),
                    types.len()
                );
            }
            "call" => {
                let inst = ins.call(callee, &[i]);
                let dfg = &ins.builder().func().dfg;
                let types = dfg
                    .inst_results(inst)
                    .iter()
                    .map(|&v| dfg.value_type(v))
                    .collect::<Vec<_>>();
                assert_eq!(types, [Type::I64, Type::I32]);
            }
            "indirect-call" => {
                let inst = ins.call_indirect(callee_sig, i, &[f]);
                assert_eq!(ins.builder().func().dfg.inst_results(inst).len(), 2);
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
        if !case.starts_with("raw-") {
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
    let data = InstructionData::from_values(Opcode::First, &[i, f]).unwrap();
    assert_eq!(
        data.result_types(&dfg, &module, &[]).unwrap().as_slice(),
        &[Type::I32]
    );
    let data = InstructionData::from_values(Opcode::First, &[unknown, i]).unwrap();
    assert!(data.result_types(&dfg, &module, &[]).is_err());
    let data = InstructionData::from_values(Opcode::Sized, &[unknown]).unwrap();
    assert_eq!(
        data.result_types(&dfg, &module, &[]).unwrap().as_slice(),
        &[Type::I8]
    );
    let output = InstructionData::Empty {
        opcode: Opcode::Output,
    };
    assert!(output.result_types(&dfg, &module, &[]).is_err());
    assert_eq!(
        output
            .result_types(&dfg, &module, &[Type::F32])
            .unwrap()
            .as_slice(),
        &[Type::F32]
    );
    let data = InstructionData::from_values(Opcode::Lane, &[i]).unwrap();
    assert!(data.result_types(&dfg, &module, &[]).is_err());
    let data = InstructionData::CallIndirect {
        sig_id: SigId(123),
        ptr: i,
        args: dfg.make_value_list(&[]),
    };
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
    builder.ins().ret(&[triple]);
    let dfg = &builder.func().dfg;
    let inst = |value| dfg.inst(dfg.value_inst(value).unwrap());
    assert!(matches!(inst(difference), InstructionData::Pair { inputs, .. } if *inputs == [a, b]));
    assert!(
        matches!(inst(reverse), InstructionData::FieldPair { right, left, .. } if (*right, *left) == (a, b))
    );
    assert!(
        matches!(inst(triple), InstructionData::Triple { args, .. } if dfg.get_value_list(*args) == [a, b, difference])
    );
    assert!(
        matches!(inst(offset), InstructionData::Immediate { value, displacement: 7, .. } if *value == ptr)
    );
    assert_eq!(dfg.value_type(selected), Type::I64);
    assert_eq!(dfg.value_type(first), Type::I32);
    assert_eq!(dfg.value_type(wide), Type::I64);
    assert_eq!(dfg.value_type(output), Type::I32);
    assert_eq!(dfg.value_type(last), Type::PTR);
    assert_eq!(dfg.value_type(first_arg), Type::I32);
    drop(builder);
    module.validate().unwrap();
    let module = module.build();
    let text = module.to_string();
    assert!(text.contains("reverse-text.i32 v1, v0"));
    assert!(!text.contains("amount="));
    veloc_mir::ModuleParser::new()
        .parse(&text)
        .unwrap()
        .validate()
        .unwrap();
    // Inferred text cannot construct mismatched result types, but callers of
    // the in-memory IR can. Check the generated validator independently too.
    let mut malformed = (*module).clone();
    malformed.functions[id].dfg.values[last].ty = Type::I32;
    malformed.functions[id].dfg.values[first_arg].ty = Type::PTR;
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
    use Constant::{Bool, I8, I32, I64};
    for (opcode, args, results, properties, expected) in [
        (
            Opcode::Direct,
            vec![I32(7), I32(3)],
            vec![Type::I32],
            vec![],
            Some(vec![I32(4)]),
        ),
        (
            Opcode::Reversed,
            vec![I32(7), I32(3)],
            vec![Type::I32],
            vec![],
            Some(vec![I32(-4)]),
        ),
        (
            Opcode::Composed,
            vec![I32(7), I32(3)],
            vec![Type::I32],
            vec![],
            Some(vec![I32(9)]),
        ),
        (
            Opcode::Trapping,
            vec![I32(7), I32(0)],
            vec![Type::I32],
            vec![],
            None,
        ),
        (
            Opcode::Trapping,
            vec![I32(7), I32(3)],
            vec![Type::I32],
            vec![],
            Some(vec![I32(2)]),
        ),
        (
            Opcode::Multiple,
            vec![I32(7), I32(7)],
            vec![Type::I32, Type::BOOL],
            vec![],
            Some(vec![I32(14), Bool(true)]),
        ),
        (
            Opcode::CompareValue,
            vec![I32(7), I32(3)],
            vec![Type::BOOL],
            vec![IntCC::GtS],
            Some(vec![Bool(true)]),
        ),
        (
            Opcode::ExtendS,
            vec![I8(-1)],
            vec![Type::I64],
            vec![],
            Some(vec![I64(-1)]),
        ),
        (Opcode::ExtendS, vec![I64(-1)], vec![Type::I8], vec![], None),
    ] {
        assert_eq!(
            evaluator::evaluate(opcode, &args, &results, &properties),
            expected,
            "{opcode:?}"
        );
    }
    assert!(!evaluator::can_fold(Opcode::VectorOnly));
    assert!(!evaluator::can_fold(Opcode::Difference));
    let compare = InstructionData::Compare {
        op: Opcode::CompareValue,
        cc: IntCC::GtS,
        args: [Value(0), Value(1)],
    };
    assert_eq!(evaluator::properties(&compare).as_slice(), &[IntCC::GtS]);
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
