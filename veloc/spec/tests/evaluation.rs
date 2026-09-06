mod common;

use common::compile_mir;

#[test]
fn offline_recipes_are_separate_and_only_emitted_on_request() {
    let output = compile_mir(
        r#"
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        op Neg<T: ScalarInteger>(arg: T) -> T {
            mnemonic: "neg", storage: Unary { arg: arg },
            semantics: bv.sub(bv.zero(), arg)
        }
        op Unmodeled<T: ScalarInteger>(arg: T) -> T {
            mnemonic: "unmodeled", storage: Unary { arg: arg }, memory: NONE
        }
    "#,
    )
    .unwrap();
    assert!(!output.opcodes.contains("Program"));
    assert!(!output.opcodes.contains("semantics: Some"));
    assert!(!output.opcodes.contains("semantics: None"));
    assert!(!output.opcodes.contains("verification"));
    assert!(output.semantics.contains("pub static SPECS:"));
    assert!(
        output
            .semantics
            .contains("veloc_semantics::SemanticSpec<veloc_mir::Opcode>")
    );
    assert!(output.semantics.contains("opcode: veloc_mir::Opcode::Neg"));
    assert!(!output.semantics.contains("Opcode::Unmodeled"));
    assert!(output.evaluation.contains("wrapping_sub"));
    assert!(!output.evaluation.contains("veloc_semantics"));
    assert!(!output.opcodes.contains("crate::semantics"));
}

#[test]
fn evaluator_specializes_only_legal_signatures_and_resolves_widths() {
    let output = compile_mir(
        r#"
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        op Widen<T: ScalarInteger>(arg: T) -> (result: ScalarInteger) {
            mnemonic: "widen", storage: Unary { arg: arg },
            where: [wider(arg, result)], semantics: bv.sext(arg, result(0))
        }
    "#,
    )
    .unwrap();
    let code = output.evaluation;
    assert_eq!(code.matches("=> {").count(), 6);
    assert!(code.contains("[Constant::I8(a0)], [Type::I64]"));
    assert!(!code.contains("[Constant::I64(a0)], [Type::I8]"));
    assert!(!code.contains("instantiate"));
    assert!(!code.contains("validate_types"));
    assert!(!code.contains("Expr"));
    assert!(!code.contains("TypeRef"));
    assert!(code.contains("<< 120"));
}

#[test]
fn vector_only_semantics_do_not_generate_scalar_evaluators() {
    let output = compile_mir(
        r#"
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        op Neg<T: vectors(ScalarInteger)>(arg: T) -> T {
            mnemonic: "neg", storage: Unary { arg: arg },
            semantics: bv.sub(bv.zero(), arg)
        }
    "#,
    )
    .unwrap();
    assert!(!output.evaluation.contains("Opcode::Neg =>"));
    assert!(
        output
            .evaluation
            .contains("can_fold(opcode: Opcode) -> bool { false }")
    );
}

#[test]
fn comparison_properties_stay_parameters_and_steps_are_shared() {
    let output = compile_mir(
        r#"
        format BinaryCond { fields: [opcode(Opcode), kind(IntCC), args(values(2))], opcode: dynamic(opcode) }
        op Test<T: ScalarInteger>(@kind: IntCC, lhs: T, rhs: T) -> T {
            mnemonic: "test", storage: BinaryCond { kind: kind, args: [lhs, rhs] },
            semantics: select(bv.cmp(kind, lhs, rhs), bv.add(lhs, rhs), bv.add(lhs, rhs))
        }
    "#,
    )
    .unwrap();
    assert_eq!(output.evaluation.matches("wrapping_add(").count(), 4);
    assert_eq!(output.evaluation.matches("match p0").count(), 4);
    assert!(output.evaluation.contains("if s2 != 0 { s3 } else { s3 }"));
}
