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
        ("type A = Type.I128;", "unknown type"),
        ("type A = I32;", "unknown type"),
        ("type A = B; type B = A;", "cyclic type definition"),
        ("type A = vector(A, 4);", "cyclic type definition"),
        ("type A = int();", "unknown type constructor"),
        ("type A = float(32, 64);", "unknown type constructor"),
        ("type A = int(7);", "unknown type constructor"),
        ("type A = float(16);", "unknown type constructor"),
        ("type A = bool(1);", "unknown type constructor"),
        ("type A = ptr(64);", "unknown type constructor"),
        (
            "type A = matrix(Type.F32, 4, 4);",
            "unknown type constructor",
        ),
        (
            "type A = vector(Type.I32);",
            "expects an element type and lane count",
        ),
        (
            "type A = vector(Type.I32, 4, true);",
            "expects an element type and lane count",
        ),
        (
            "type A = vector(vector(Type.I32, 4), 4);",
            "must be a scalar type",
        ),
        ("type A = vector(Type.PTR, 4);", "invalid vector"),
        ("type A = vector(Type.I32, scalable());", "vector shape"),
        ("type A = vector(Type.I32, scalable(3));", "vector lanes"),
        (
            "type A = vector(Type.I32, scalable(2147483648));",
            "supported type domain",
        ),
        (
            "type A = Type.I32 | Type.I64;",
            "expected a type name or type constructor",
        ),
        ("type A = Integer;", "unknown type"),
        ("type lower = Type.I32;", "uppercase"),
        ("type INVALID = Type.I32;", "not INVALID"),
        ("type A = Type.I32; type A = Type.I64;", "duplicate type"),
        ("type A = Type.I32", "expected `;`"),
        (
            "scalar A { code: 9, kind: integer, bits: 32 }",
            "unknown definition kind",
        ),
        (
            "vector A { element: Type.I32, lanes: 4 }",
            "unknown definition kind",
        ),
    ] {
        rejected(&common::source(source), message);
    }
}

#[test]
fn deeply_nested_type_construction_is_bounded_by_the_parser() {
    let source = format!(
        "type A = {}Type.I32{};",
        "vector(".repeat(70),
        ", 4)".repeat(70)
    );
    rejected(&common::source(&source), "nesting exceeds 64");
}
