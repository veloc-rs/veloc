mod common;

use common::{BUILTINS, compile};

const ADD: &str = r#"
record Pair { args: values(2) }
op Add<T: Bits>(lhs: T, rhs: T) -> (result: T) {
    meta: OpInfo {},
    mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
}
class Bits { members: [ScalarInteger] }
"#;

fn rejected(source: &str, expected: &str) {
    let result = std::panic::catch_unwind(|| veloc_opgen::compile(source))
        .expect("invalid definitions must produce diagnostics, not panic");
    let error = result.err().expect("definition should be rejected");
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn builtin_references_are_explicit_not_hidden_mir_defaults() {
    let encoding = "encoding Type { storage: u16, fields: [scalar(4), lanes_log2(4), scalable(1)], codes: [] }";
    rejected(
        &format!("{encoding}\n{ADD}"),
        "unknown type or class `ScalarInteger`",
    );
    rejected(
        &format!(
            "{}\nop Test() -> (result: I32) {{ mnemonic: \"test\" }}",
            encoding
        ),
        "unbound type variable `I32`",
    );
}

#[test]
fn class_unions_drive_both_generated_contracts_and_semantic_checks() {
    let output = compile(ADD).unwrap();
    assert!(output.type_rules.contains("C::Bits"));
    assert!(output.opcodes.contains("1..=4 => 0x00000001,"));
    let mixed = ADD.replace(
        "members: [ScalarInteger]",
        "members: [ScalarInteger, ScalarFloat]",
    );
    rejected(
        &common::source(&mixed),
        "floating-point execution semantics are not modeled",
    );
    let vectors = ADD.replace("members: [ScalarInteger]", "members: [Integer & Vector]");
    assert!(compile(&vectors).is_ok());
}

#[test]
fn class_domains_check_derived_shapes_without_canonical_class_names() {
    let source = r#"
        record Unary { arg: Value }
        class Lanes { members: [ScalarInteger] }
        op Element<T: Lanes>(arg: T) -> (result: element(T)) {
    meta: OpInfo { memory: Known([]) },
            mnemonic: "element", storage: Unary { arg: arg }, }
    "#;
    rejected(&common::source(source), "impossible element constraint");
    assert!(compile(&source.replace("[ScalarInteger]", "[vectors(ScalarInteger)]")).is_ok());
}

#[test]
fn floating_text_uses_domains_instead_of_class_name_allowlists() {
    let source = r#"
        class Floating { members: [ScalarFloat] }
        record Literal { value: Float }
        op Literal(@value: Float) -> (result: Floating) {
    meta: OpInfo { memory: Known([]) },
            mnemonic: "literal", storage: Literal { value: value },
            }
    "#;
    assert!(compile(source).is_ok());
    rejected(
        &common::source(&source.replace("[ScalarFloat]", "[vectors(ScalarFloat)]")),
        "scalar float",
    );
}

#[test]
fn classes_reject_cycles_unknowns_duplicates_and_shadowing() {
    for (defs, error) in [
        (
            "class A { members: [B] } class B { members: [A] }",
            "cyclic type class",
        ),
        ("class A { members: [A] }", "cyclic type class"),
        ("class A { members: [Absent] }", "unknown type or class"),
        ("class A { members: [] }", "must not be empty"),
        (
            "class A { members: [Scalar, Scalar] }",
            "duplicate class member",
        ),
        (
            "class A { members: [scalar_integer] }",
            "unknown type or class",
        ),
        (
            "class values { members: [Scalar] }",
            "shadows a signature keyword",
        ),
        ("class I32 { members: [Scalar] }", "shadows an exact type"),
        (
            "class Scalar { members: [scalar_integer] }",
            "duplicate class",
        ),
        ("class A { members: [Scalar], typo: 1 }", "unknown field"),
    ] {
        rejected(&common::source(defs), error);
    }
}

#[test]
fn compact_scalar_codes_and_adapter_contracts_are_checked() {
    for (from, to, error) in [
        ("I8(1)", "I8(0)", "scalar code"),
        ("I8(1)", "I8(16)", "scalar code"),
        ("I8(1)", "I8(2)", "scalar code"),
        ("int(8)", "int(7)", "unsupported scalar"),
        ("float(32)", "float(target)", "expected a number"),
        ("ptr()", "ptr(64)", "expects no arguments"),
        ("type I8", "type BYTE", "unknown type `I8`"),
        ("int(8)", "int(16)", "requires name `I16`"),
    ] {
        rejected(&BUILTINS.replace(from, to), error);
    }
    let output = veloc_opgen::compile(BUILTINS).unwrap();
    assert!(output.types.contains("I8 = 1,"));
    assert!(output.types.contains("PTR = 8,"));
    assert!(output.types.contains("pub const BOOL: Self"));
    assert!(output.types.contains("\"bool\" => Some(Self::BOOL)"));
}

#[test]
fn effect_sets_use_the_declared_vocabulary() {
    assert!(compile(&ADD.replace("OpInfo {}", "OpInfo { memory: Known([]) }")).is_ok());
    for value in [
        "Known([READ])",
        "Known([WRITE])",
        "Known([ALLOCATE])",
        "Known([FREE])",
        "Unknown",
    ] {
        rejected(
            &common::source(&ADD.replace("OpInfo {}", &format!("OpInfo {{ memory: {value} }}"))),
            "no memory effects or control flow",
        );
    }
}

#[test]
fn inferred_traits_must_also_be_declared() {
    let source = common::source(ADD).replace("COMMUTATIVE(1), ", "");
    rejected(
        &source,
        "semantic law requires undeclared trait `COMMUTATIVE`",
    );
}

#[test]
fn builtin_diagnostics_keep_the_original_member_location() {
    let source = "\n\n// diagnostics retain their source location\n\nflags Bad { storage: u8, members: [LOW(0), HIGH(0)], separator: \",\" }";
    let error = veloc_opgen::parse(&format!("{source}\n{}", common::TYPES))
        .err()
        .unwrap();
    assert_eq!(error.line, 5);
    assert_eq!(error.column, 44); // HIGH(0) overlaps LOW(0).
}
