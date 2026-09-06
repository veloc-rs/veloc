mod common;

use common::{BUILTINS, compile_mir};

fn rejected(source: &str, expected: &str) {
    let result =
        std::panic::catch_unwind(|| compile_mir(source)).expect("invalid types must not panic");
    let error = result.err().expect("invalid types were accepted");
    assert!(error.message.contains(expected), "{error}");
}

#[test]
fn exact_class_members_drive_codegen_and_bitvector_semantics() {
    let source = r#"
        class Wide { members: [I32, I64] }
        format Pair { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
        op Add<T: Wide>(lhs: T, rhs: T) -> T {
            mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.builtins.contains("3 | 4 => 0x00000001,\n_ => 0,"));
    assert!(output.types.contains("C::Wide.accepts(operands[0])"));
    rejected(
        &source.replace("[I32, I64]", "[I32, F64]"),
        "floating-point execution semantics are not modeled",
    );
    assert!(compile_mir(&source.replace("[I32, I64]", "[I32X4]")).is_ok());
}

#[test]
fn vector_declarations_generate_constants_and_exact_type_patterns() {
    let output = compile_mir(
        r#"
        class Chosen { members: [I32X4, SV4] }
        type SV4 = vector(I32, scalable(4));
        type MV8 = vector(BOOL, 8);
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        op Copy(arg: SV4) -> SV4 { mnemonic: "copy", storage: Unary { arg: arg }, memory: NONE }
    "#,
    )
    .unwrap();
    assert!(
        output
            .scalars
            .contains("pub const SV4: Self = Self::I32.as_scalar().expect(\"checked scalar definition\").vector(4, true)")
    );
    assert!(
        output
            .scalars
            .contains("pub const MV8: Self = Self::BOOL.as_scalar().expect(\"checked scalar definition\").vector(8, false)")
    );
    assert!(output.types.contains("operands[0] == Type::SV4"));
    assert!(output.builtins.contains("3 => 0x00040004,"));
}

#[test]
fn exact_shapes_detect_impossible_relations_at_definition_time() {
    let source = r#"
        class V4 { members: [I32X4] }
        class V2 { members: [I64X2] }
        format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
        op Convert<T: V4>(arg: T) -> shape(T, V2) {
            mnemonic: "convert", storage: Unary { arg: arg }, memory: NONE
        }
    "#;
    rejected(source, "impossible shape constraint");
    assert!(compile_mir(&source.replace("[I64X2]", "[F32X4]")).is_ok());
    let scalar = source.replace("[I64X2]", "[I64]");
    rejected(&scalar, "impossible shape constraint");
}

#[test]
fn vector_families_require_scalar_sets_and_preserve_definition_checks() {
    for (source, error) in [
        (
            "class Bad { members: [vectors(PTR)] }",
            "non-pointer scalar",
        ),
        (
            "class Bad { members: [vectors(I32X4)] }",
            "non-pointer scalar",
        ),
        (
            "class Bad { members: [vectors(Any)] }",
            "non-pointer scalar",
        ),
        (
            "class Bad { members: [vectors(vectors(I32))] }",
            "non-pointer scalar",
        ),
        (
            "class Bad { members: [vectors(Missing)] }",
            "unknown type or class",
        ),
        (
            "class Bad { members: [vectors()] }",
            "expected a type, class or vectors(set)",
        ),
        (
            "class Bad { members: [vectors(I32, I64)] }",
            "expected a type, class or vectors(set)",
        ),
        (
            "class Bad { members: [vectors(I32), vectors(I32)] }",
            "duplicate class member",
        ),
        (
            "class A { members: [vectors(B)] } class B { members: [A] }",
            "cyclic type class",
        ),
        ("class I32X4 { members: [I32] }", "shadows an exact type"),
        (
            "class Bad { members: [vector_integer] }",
            "unknown type or class",
        ),
    ] {
        rejected(source, error);
    }
    assert!(compile_mir("class Mixed { members: [I32, ScalarInteger] }").is_ok());
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
    let error = veloc_opgen::compile_mir(&format!("{narrow}\ntype V = vector(I32, 256);"))
        .err()
        .unwrap();
    assert!(error.message.contains("vector lanes"));
}
