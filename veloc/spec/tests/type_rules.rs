mod common;

#[test]
fn equal_contracts_share_handlers_without_opcode_name_knowledge() {
    let source = r#"
        format Binary { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
        op First<T: Integer>(lhs: T, rhs: T) -> T {
            mnemonic: "first", storage: Binary { args: [lhs, rhs] }, memory: NONE
        }
        op Second<U: Integer>(a: U, b: U) -> U {
            mnemonic: "second", storage: Binary { args: [a, b] }, memory: NONE
        }
    "#;
    let output = common::compile_mir(source).unwrap();
    assert_eq!(output.types.matches("fn validate_0(").count(), 1);
    assert!(!output.types.contains("fn validate_1("));
    let inference = output.types.split("fn infer_0(").nth(1).unwrap();
    assert!(inference.contains("let results = [operands[0]];"));
    assert!(!inference.contains("results[0]"));
    assert!(!inference.contains("results.len()"));
    assert!(
        output
            .opcodes
            .contains("Self::First | Self::Second => crate::opspec::type_rules::validate_0")
    );
    let output = common::compile_mir(&source.replace("U: Integer", "U: Float")).unwrap();
    assert!(output.types.contains("fn validate_1("));
    assert!(!output.opcodes.contains("Self::First | Self::Second"));
}

#[test]
fn different_relations_keep_separate_handlers() {
    let output = common::compile_mir(
        r#"
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        op Widen<T: Integer>(arg: T) -> (result: shape(T, Integer)) {
            mnemonic: "widen",
            storage: Unary { arg: arg }, memory: NONE,
            where: [wider(arg, result)]
        }
        op Narrow<T: Integer>(arg: T) -> (result: shape(T, Integer)) {
            mnemonic: "narrow",
            storage: Unary { arg: arg }, memory: NONE,
            where: [narrower(arg, result)]
        }
        op Unrestricted<T: Integer>(arg: T) -> shape(T, Integer) {
            mnemonic: "unrestricted",
            storage: Unary { arg: arg }, memory: NONE
        }
        "#,
    )
    .unwrap();
    assert_eq!(output.types.matches("fn validate_").count(), 3);
    assert!(output.types.contains("to > from"));
    assert!(output.types.contains("to < from"));
}

#[test]
fn all_contracts_are_compiled_without_an_interpreter_fallback() {
    let output = common::compile_mir(include_str!("../../mir/tests/defs/type_rules.ops")).unwrap();
    assert!(!output.types.contains("static SCHEME_"));
    assert_eq!(output.types.matches("fn validate_").count(), 3);
    assert_eq!(output.types.matches("fn infer_").count(), 3);
    assert!(
        output
            .types
            .contains("let results = [operands[4], operands[0]];")
    );
    assert!(output.types.contains("Ok(super::ResultTypes::Explicit)"));
    assert!(output.types.contains(
        "TypeError::Relation(\"results[0] must have more bits per lane than operands[0]\")"
    ));
    assert!(!output.types.contains("TypeRelation"));
    assert!(!output.types.contains("TypeSlot"));
    assert!(!output.types.contains("TypePattern"));
    assert!(!output.types.contains("TypeList"));
    assert!(!output.types.contains("relations:"));
    assert!(!output.types.contains("bindings"));
    let dispatch = output
        .opcodes
        .split("pub fn validate_types(")
        .nth(1)
        .unwrap()
        .split("pub fn infer_result_types(")
        .next()
        .unwrap();
    assert!(!dispatch.contains("_ =>"));
    assert!(!output.opcodes.contains("specialized-types"));
    assert!(!output.opcodes.contains("self.spec().type_scheme.validate"));
}
