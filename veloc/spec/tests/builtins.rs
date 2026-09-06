mod common;

use common::{BUILTINS, compile_mir};

const ADD: &str = r#"
format Pair { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
op Add<T: Bits>(lhs: T, rhs: T) -> (result: T) {
    mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
}
class Bits { members: [ScalarInteger] }
"#;

fn rejected(source: &str, expected: &str) {
    let result = std::panic::catch_unwind(|| veloc_opgen::compile_mir(source))
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
    let output = compile_mir(ADD).unwrap();
    assert!(output.types.contains("C::Bits"));
    assert!(output.builtins.contains("1..=4 => 0x00000001,"));
    let mixed = ADD.replace(
        "members: [ScalarInteger]",
        "members: [ScalarInteger, ScalarFloat]",
    );
    rejected(
        &common::source(&mixed),
        "floating-point execution semantics are not modeled",
    );
    let vectors = ADD.replace("members: [ScalarInteger]", "members: [Integer & Vector]");
    assert!(compile_mir(&vectors).is_ok());
}

#[test]
fn class_domains_check_derived_shapes_without_canonical_class_names() {
    let source = r#"
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        class Lanes { members: [ScalarInteger] }
        op Element<T: Lanes>(arg: T) -> (result: element(T)) {
            mnemonic: "element", storage: Unary { arg: arg }, memory: NONE
        }
    "#;
    rejected(&common::source(source), "impossible element constraint");
    assert!(compile_mir(&source.replace("[ScalarInteger]", "[vectors(ScalarInteger)]")).is_ok());
}

#[test]
fn floating_text_uses_domains_instead_of_class_name_allowlists() {
    let source = r#"
        class Floating { members: [ScalarFloat] }
        format Literal { fields: [opcode(Opcode), value(u64)], opcode: dynamic(opcode) }
        op Literal(@value: u64) -> (result: Floating) {
            mnemonic: "literal", storage: Literal { value: value },
            text: Text { args: [float(value)] }, memory: NONE
        }
    "#;
    assert!(compile_mir(source).is_ok());
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
    let output = veloc_opgen::compile_mir(BUILTINS).unwrap();
    assert!(
        output
            .scalars
            .contains("pub const I8: Self = Self(1 << SCALAR_SHIFT);")
    );
    assert!(
        output
            .scalars
            .contains("pub const PTR: Self = Self(8 << SCALAR_SHIFT);")
    );
    assert!(output.scalars.contains("pub const BOOL: Self"));
    assert!(output.scalars.contains("\"bool\" => Some(Self::BOOL)"));
}

#[test]
fn traits_and_regions_have_checked_bits_and_generated_names() {
    for (defs, error) in [
        ("trait EXTRA { bit: 0 }", "flag bit must be unique"),
        ("trait EXTRA { bit: 16 }", "less than 16"),
        ("region EXTRA { bit: 0 }", "flag bit must be unique"),
        ("region EXTRA { bit: 8 }", "less than 8"),
        ("trait NAMES { bit: 5 }", "cannot be NONE, ALL or NAMES"),
        ("region ALL { bit: 5 }", "cannot be NONE, ALL or NAMES"),
        ("trait lower { bit: 5 }", "must be uppercase"),
    ] {
        rejected(&common::source(defs), error);
    }
    let output = compile_mir("trait EXTRA_FACT { bit: 5 } region DEVICE { bit: 5 }").unwrap();
    assert!(
        output
            .builtins
            .contains("pub const EXTRA_FACT: Self = Self(1 << 5)")
    );
    assert!(
        output
            .builtins
            .contains("(Self::EXTRA_FACT, \"extra-fact\")")
    );
    assert!(output.builtins.contains("pub const ALL: Self = Self(63)"));
    assert!(
        output
            .builtins
            .contains("pub const UNKNOWN: Self = Self::new(MemoryRegions(63), MemoryRegions(63))")
    );
}

#[test]
fn effects_use_declared_regions_and_purity_not_effect_names() {
    let pure = format!(
        "effect PURE {{ reads: [], writes: [] }}\n{}",
        ADD.replace("semantics:", "memory: PURE, semantics:")
    );
    assert!(compile_mir(&pure).is_ok());
    rejected(
        &common::source(&pure.replace("reads: []", "reads: [HEAP]")),
        "no memory effects or control flow",
    );
    for (defs, error) in [
        (
            "effect BAD { reads: [MISSING], writes: [] }",
            "unknown memory region",
        ),
        (
            "effect BAD { reads: [HEAP, HEAP], writes: [] }",
            "duplicate memory region",
        ),
        (
            "effect BAD { reads: [ALL, HEAP], writes: [] }",
            "ALL must be the only region",
        ),
    ] {
        rejected(&common::source(defs), error);
    }
    rejected(
        &BUILTINS.replace("effect NONE { reads: []", "effect NONE { reads: [HEAP]"),
        "NONE must have no memory effects",
    );
    rejected(
        &BUILTINS.replace("reads: [ALL]", "reads: [HEAP]"),
        "UNKNOWN must read and write all regions",
    );
    let output = veloc_opgen::compile_mir(BUILTINS).unwrap();
    for (name, read, write) in [
        ("GLOBAL_READ", 4, 0),
        ("GLOBAL_WRITE", 0, 4),
        ("TABLE_READ", 8, 0),
        ("TABLE_WRITE", 0, 8),
    ] {
        assert!(output.builtins.contains(&format!(
            "pub const {name}: Self = Self::new(MemoryRegions({read}), MemoryRegions({write}))"
        )));
    }
}

#[test]
fn inferred_traits_must_also_be_declared() {
    let source = common::source(ADD).replace("trait COMMUTATIVE { bit: 1 }", "");
    rejected(
        &source,
        "semantic law requires undeclared trait `COMMUTATIVE`",
    );
}

#[test]
fn builtin_diagnostics_keep_the_original_record_location() {
    let source = "\n\nregion HEAP { bit: 0 }\n\neffect NONE { reads: [HEAP], writes: [] }";
    let error = veloc_opgen::parse(&format!("{source}\n{}", common::TYPES))
        .err()
        .unwrap();
    assert_eq!(error.line, 5);
    assert_eq!(error.column, 1);
}
