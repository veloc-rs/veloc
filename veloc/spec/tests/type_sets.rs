mod common;

use common::{BUILTINS, compile};

fn rejected(source: &str, expected: &str) {
    let result =
        std::panic::catch_unwind(|| compile(source)).expect("invalid types must not panic");
    let error = result.err().expect("invalid types were accepted");
    assert!(error.message.contains(expected), "{error}");
}

#[test]
fn exact_sets_drive_codegen_and_bitvector_semantics() {
    let source = r#"
        typeset Wide = I32 | I64;
        struct Pair { args: values(2) }
        op Add<T: Wide>(lhs: T, rhs: T) -> T {
    meta: OpInfo {},
            mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
        }
    "#;
    compile(source).unwrap();

    rejected(
        &source.replace("I32 | I64", "I32 | F64"),
        "floating-point execution semantics are not modeled",
    );
    assert!(compile(&source.replace("I32 | I64", "I32X4")).is_ok());
}

#[test]
fn exact_shapes_detect_impossible_relations_at_definition_time() {
    let source = r#"
        typeset V4 = I32X4;
        typeset V2 = I64X2;
        struct Unary { arg: Value }
        op Convert<T: V4, U: V2>(arg: T) -> U {
    verify { require(U.same_shape(T), "input and result must have the same shape"); }
    meta: OpInfo { memory: Known([]) },
            mnemonic: "convert", storage: Unary { arg: arg }, }
    "#;
    rejected(source, "constraint is always false");
    assert!(compile(&source.replace("I64X2", "F32X4")).is_ok());
    let scalar = source.replace("I64X2", "I64");
    rejected(&scalar, "constraint is always false");
}

#[test]
fn vector_families_require_scalar_sets_and_preserve_definition_checks() {
    for (source, error) in [
        ("typeset Bad = vectors(PTR);", "non-pointer scalar"),
        ("typeset Bad = vectors(I32X4);", "non-pointer scalar"),
        ("typeset Bad = vectors(Any);", "non-pointer scalar"),
        ("typeset Bad = vectors(vectors(I32));", "non-pointer scalar"),
        ("typeset Bad = vectors(Missing);", "unknown type or typeset"),
        (
            "typeset Bad = vectors();",
            "expected a type, typeset or vectors(set)",
        ),
        (
            "typeset Bad = vectors(I32, I64);",
            "expected a type, typeset or vectors(set)",
        ),
        ("typeset A = vectors(B); typeset B = A;", "cyclic type set"),
        ("typeset I32X4 = I32;", "shadows an exact type"),
        ("typeset Bad = vector_integer;", "unknown type or typeset"),
    ] {
        rejected(source, error);
    }
    assert!(compile("typeset Mixed = I32 | ScalarInteger;").is_ok());
}

#[test]
fn vector_constants_reject_unrepresentable_or_invalid_types() {
    for (source, error) in [
        ("type V = vector(PTR, 4);", "pointer vectors"),
        ("type V = vector(I32X4, 4);", "must be a scalar type"),
        ("type V = vector(Missing, 4);", "unknown type"),
        ("type V = vector(I32, 0);", "vector lanes"),
        ("type V = vector(I32, 1);", "vector lanes"),
        ("type V = vector(I32, 3);", "vector lanes"),
        ("type V = vector(I32, 65536);", "vector lanes"),
        ("type V = vector(I32, maybe);", "vector shape must be"),
        ("type I32 = vector(I32, 4);", "duplicate type"),
        ("type INVALID = vector(I32, 4);", "not INVALID"),
        ("type I32X4 = vector(I32, 4);", "duplicate type"),
    ] {
        rejected(source, error);
    }
    let narrow = BUILTINS.replace("lanes_log2(4)", "lanes_log2(3)");
    let error = veloc_opgen::compile(&format!("{narrow}\ntype V = vector(I32, 256);"))
        .err()
        .unwrap();
    assert!(error.message.contains("vector lanes"));
}
