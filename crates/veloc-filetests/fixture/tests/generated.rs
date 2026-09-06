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
