mod common;

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
            "supported type domain",
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
fn deeply_nested_type_construction_is_bounded_by_the_parser() {
    let source = format!("type A = {}I32{};", "vector(".repeat(70), ", 4)".repeat(70));
    rejected(&common::source(&source), "nesting exceeds 64");
}
