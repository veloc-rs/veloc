mod common;

use common::BUILTINS;

fn rejected(source: &str, message: &str) {
    let error = std::panic::catch_unwind(|| veloc_opgen::compile(source))
        .expect("bad type declarations must not panic")
        .err()
        .expect("bad type declaration was accepted");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn malformed_constructors_alias_cycles_and_removed_syntax_are_rejected() {
    for (source, message) in [
        ("type A = Missing;", "unknown type"),
        ("type A = B; type B = A;", "cyclic type definition"),
        ("type A = vector(A, 4);", "cyclic type definition"),
        ("type A = int();", "expects one bit width"),
        ("type A = float(32, 64);", "expects one bit width"),
        ("type A = int(7);", "unsupported scalar"),
        ("type A = float(16);", "unsupported scalar"),
        ("type A = bool(1);", "expects no arguments"),
        ("type A = ptr(64);", "expects no arguments"),
        ("type A = matrix(F32, 4, 4);", "unknown type constructor"),
        (
            "type A = vector(I32);",
            "expects an element type and lane count",
        ),
        (
            "type A = vector(I32, 4, true);",
            "expects an element type and lane count",
        ),
        (
            "type A = vector(vector(I32, 4), 4);",
            "must be a scalar type",
        ),
        ("type A = vector(ptr(), 4);", "pointer vectors"),
        ("type A = vector(I32, scalable());", "vector shape"),
        ("type A = vector(I32, scalable(3));", "vector lanes"),
        (
            "type A = vector(I32, scalable(2147483648));",
            "fit Type encoding",
        ),
        (
            "type A = I32 | I64;",
            "expected a type name or type constructor",
        ),
        ("type A = Integer;", "unknown type"),
        ("type lower = I32;", "uppercase"),
        ("type INVALID = I32;", "not INVALID"),
        ("type I32 = int(32);", "duplicate type"),
        ("type A = I32", "expected `;`"),
        (
            "scalar A { code: 9, kind: integer, bits: 32 }",
            "unknown definition kind",
        ),
        (
            "vector A { element: I32, lanes: 4 }",
            "unknown definition kind",
        ),
    ] {
        rejected(&common::source(source), message);
    }
}

#[test]
fn encoding_bindings_are_separate_checked_and_not_assigned_to_aliases() {
    for (from, to, message) in [
        ("I8(1), ", "", "missing scalar encoding for `I8`"),
        ("I8(1)", "MISSING(1)", "encoding references unknown type"),
        ("I8(1)", "I32X4(1)", "cannot assign a code to a vector"),
        ("I8(1)", "I8(1), I8(9)", "duplicate scalar encoding"),
        ("I8(1)", "I8(0)", "scalar code"),
        ("I8(1)", "I8(16)", "scalar code"),
        ("I8(1)", "I8(2)", "scalar code"),
        ("I8(1)", "I8", "expected scalar encoding"),
        ("I8(1)", "I8()", "expected scalar encoding"),
        ("I8(1)", "I8(1, 2)", "expected scalar encoding"),
        ("I8(1)", "I8(code)", "code must be a number"),
        ("codes:", "unknown_codes:", "missing `codes`"),
    ] {
        rejected(&BUILTINS.replace(from, to), message);
    }
    let alias = format!("{}\ntype BYTE = I8;", BUILTINS.replace("I8(1)", "BYTE(1)"));
    rejected(&alias, "requires name `I8`");

    veloc_opgen::compile(&BUILTINS.replace("I32(3)", "I32(9)")).unwrap();
}

#[test]
fn deeply_nested_type_construction_is_bounded_by_the_parser() {
    let source = format!("type A = {}I32{};", "vector(".repeat(70), ", 4)".repeat(70));
    rejected(&common::source(&source), "nesting exceeds 64");
}
