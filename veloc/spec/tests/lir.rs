mod common;
use veloc_semantics::BvOp;

const BINARY: &str =
    "storage Operands { prefix: \"G_\" }\nformat Binary { fields: [dst(Def), lhs(Use), rhs(Use)] }";
const ADD: &str =
    "op G_SUM<T: Integer>(lhs: T, rhs: T) -> T { storage: Binary, semantics: bv.add(lhs, rhs) }";

#[test]
fn shared_operation_model_drives_storage_and_lowering() {
    let source = common::source(&format!("{BINARY}\n{ADD}"));
    let definitions = veloc_opgen::parse(&source).unwrap();
    assert_eq!(definitions.primitive_bindings(), [(BvOp::Add, "G_SUM")]);
    assert_eq!(definitions.operation_count(), 1);
    assert_eq!(definitions.format_count(), 1);
    let generated = veloc_opgen::compile(&source).unwrap();
    assert!(
        generated
            .instructions
            .contains("pub fn build_sum(dst: Writable<Reg>, lhs: Reg, rhs: Reg)")
    );
    assert!(
        generated
            .instructions
            .contains("GenericOpcode::G_SUM => Self::Binary")
    );
    assert!(generated.instructions.contains("pub fn as_binary(&self)"));
    assert!(generated.type_rules.contains("impl crate::GenericOpcode"));
}

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
            "storage Operands {} format Bad { fields: [dst(Def), dst(Use)] }".into(),
            "duplicate field",
        ),
        (
            "storage Operands {} format Bad { fields: [values(Uses), dst(Def)] }".into(),
            "entire operand sequence",
        ),
        (
            "storage Operands {} format Bad { fields: [dst(Def)], lengths: [0] }".into(),
            "operand counts",
        ),
        (
            "storage Operands {} format Bad { fields: [dst(Def)], lengths: [1, 1] }".into(),
            "operand counts",
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

#[test]
fn complete_contracts_retain_reviewed_primitive_set() {
    let defs =
        veloc_opgen::parse(&common::source(include_str!("../../lir/defs/generic.ops"))).unwrap();
    let mut primitives = defs.primitive_bindings();
    primitives.sort_by_key(|(_, name)| *name);
    assert_eq!(
        primitives,
        [
            (BvOp::Add, "G_ADD"),
            (BvOp::And, "G_AND"),
            (BvOp::Mul, "G_MUL"),
            (BvOp::Neg, "G_NEG"),
            (BvOp::Or, "G_OR"),
            (BvOp::Sub, "G_SUB"),
            (BvOp::Xor, "G_XOR"),
        ]
    );
}
