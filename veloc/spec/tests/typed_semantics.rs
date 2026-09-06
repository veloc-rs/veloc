mod common;
use common::compile_mir;

#[test]
fn traps_are_typed_behavior_not_input_assumptions() {
    let source = r#"
        format Binary { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
        op Divide<T: ScalarInteger>(lhs: T, rhs: T) -> T {
            mnemonic: "divide", storage: Binary { args: [lhs, rhs] },
            semantics: bv.udiv(lhs, rhs),
            traps: [DivisionByZero(bv.eq(rhs, bv.zero()))]
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.opcodes.contains("Trap::DivisionByZero"));
    assert!(output.opcodes.contains("MAY_TRAP"));
    rejected(
        &source.replace("bv.eq(rhs, bv.zero())", "rhs"),
        "invalid semantic types",
    );
    rejected(
        &source.replace("DivisionByZero", "UnknownTrap"),
        "unknown trap",
    );
    rejected(
        &source.replace("semantics: bv.udiv(lhs, rhs),", ""),
        "trap guards require executable semantics",
    );
}

const CONVERT: &str = r#"
format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
op Convert<T: Integer>(arg: T) -> (result: shape(T, Integer)) {
    mnemonic: "convert", storage: Unary { arg: arg },
    where: [wider(arg, result)], semantics: bv.sext(arg, result(0))
}
"#;

fn rejected(source: &str, expected: &str) {
    let error = compile_mir(source)
        .err()
        .expect("invalid semantics accepted");
    assert!(error.message.contains(expected), "{error}");
}

#[test]
fn widths_are_checked_against_every_signature_instance() {
    assert!(compile_mir(CONVERT).is_ok());
    rejected(
        &CONVERT.replace("wider(arg, result)", "narrower(arg, result)"),
        "cannot extend",
    );
    rejected(&CONVERT.replace("result(0)", "result(1)"), "results");
    rejected(
        &CONVERT.replace("result(0)", "type(missing)"),
        "expected type(operand)",
    );
    rejected(
        &CONVERT.replace("bv.sext(arg, result(0))", "arg"),
        "invalid semantic types",
    );
    rejected(
        &CONVERT.replace("shape(T, Integer)", "I32X4"),
        "shared lane shape",
    );
}

#[test]
fn multiple_results_require_matching_sorts_and_counts() {
    let source = r#"
        format Binary { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
        op Pair<T: ScalarInteger>(lhs: T, rhs: T) -> (T, BOOL) {
            mnemonic: "pair", storage: Binary { args: [lhs, rhs] },
            semantics: [bv.add(lhs, rhs), bv.ult(lhs, rhs)]
        }
    "#;
    assert!(compile_mir(source).is_ok());
    rejected(
        &source.replace("bv.ult(lhs, rhs)", "bv.add(lhs, rhs)"),
        "invalid semantic types",
    );
    rejected(
        &source.replace("[bv.add(lhs, rhs), bv.ult(lhs, rhs)]", "bv.add(lhs, rhs)"),
        "result count",
    );
}

#[test]
fn comparison_properties_are_bound_by_name_not_ssa_position() {
    let source = r#"
        format Compare { fields: [opcode(Opcode), cc(IntCC), args(values(2))], opcode: dynamic(opcode) }
        op Test<T: ScalarInteger>(lhs: T, @condition: IntCC, rhs: T) -> BOOL {
            mnemonic: "test", storage: Compare { cc: condition, args: [lhs, rhs] },
            semantics: bv.cmp(condition, lhs, rhs)
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.opcodes.contains("ComparisonRef::Property(0)"));
    assert!(output.opcodes.contains("cc.predicate()"));
    rejected(
        &source.replace("bv.cmp(condition", "bv.cmp(missing"),
        "unknown integer comparison property",
    );
    rejected(
        &source.replace("bv.cmp(condition, lhs, rhs)", "bv.add(condition, lhs)"),
        "unknown semantic value",
    );
}
