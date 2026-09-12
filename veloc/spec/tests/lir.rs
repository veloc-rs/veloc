mod common;

const BINARY: &str =
    "storage Operands { prefix: \"G_\" }\nstruct Binary { dst: Def, lhs: Use, rhs: Use }";
const ADD: &str = "op G_SUM<T: Integer>(lhs: T, rhs: T) -> T { meta: OpInfo {}, storage: Binary, semantics: bv.add(lhs, rhs) }";

#[test]
fn invalid_definitions_fail_before_emission() {
    for (source, message) in [
        (
            format!(
                "{BINARY} {}",
                ADD.replace("storage: Binary", "storage: Missing")
            ),
            "unknown operand format",
        ),
        (
            format!("{BINARY} {}", ADD.replace("rhs: T", "other: T")),
            "logical signature",
        ),
        (
            format!("{BINARY} {}", ADD.replace("-> T", "-> (T, T)")),
            "logical signature",
        ),
        (
            format!("{BINARY} {}", ADD.replace("Integer", "Missing")),
            "Missing",
        ),
        (
            format!("{BINARY} {}", ADD.replace("rhs)", "missing)")),
            "unknown semantic value",
        ),
        (
            format!("{BINARY} {}", ADD.replace("Integer", "Float")),
            "floating-point",
        ),
        (
            format!(
                "{BINARY} {}",
                ADD.replace("semantics:", "flow: Call, semantics:")
            ),
            "MAY_TRAP",
        ),
        (format!("{BINARY} {ADD} {ADD}"), "duplicate op"),
        (
            "storage Operands {} struct Bad { dst: Def, dst: Use }".into(),
            "duplicate field",
        ),
        (
            "storage Operands {} struct Bad { values: Uses, dst: Def }".into(),
            "entire operand sequence",
        ),
        (
            "storage Operands {} struct Bad { dst: Def } layout Bad { lengths: [0] }".into(),
            "unknown field",
        ),
        (
            "storage Operands {} struct Bad { dst: Def } layout Bad { lengths: [1, 1] }".into(),
            "unknown field",
        ),
        (
            format!(
                "{BINARY} {}",
                ADD.replace("semantics:", "arity: 3, semantics:")
            ),
            "arity",
        ),
    ] {
        let error = veloc_opgen::parse(&common::source(&source))
            .err()
            .expect(&source);
        assert!(error.message.contains(message), "{source}\n{error}");
    }
}

#[test]
fn unsupported_output_contracts_are_not_silently_ignored() {
    let source = common::source(&format!(
        "{BINARY} {}",
        ADD.replace(
            "semantics:",
            "constraints: [require(true, \"checked\")], semantics:"
        )
    ));
    assert!(
        veloc_opgen::compile(&source)
            .err()
            .unwrap()
            .message
            .contains("does not yet support")
    );
}
