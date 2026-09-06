mod common;

use common::{BUILTINS, compile_mir};

fn rejected(source: &str, message: &str) {
    let error = std::panic::catch_unwind(|| veloc_opgen::compile_mir(source))
        .expect("bad type declarations must not panic")
        .err()
        .expect("bad type declaration was accepted");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn type_expressions_support_aliases_forward_references_and_nested_construction() {
    let source = r#"
        type WORDS = vector(WORD, 4);
        type MASK = vector(bool(), scalable(8));
        type WORD = LATER;
        type LATER = I32;
        type COPY = WORDS;
        class Chosen { members: [WORD, WORDS] }
        predicate is_chosen = Chosen;
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.scalars.contains("pub const WORD: Self = Self::I32;"));
    assert!(
        output
            .scalars
            .contains("pub const WORDS: Self = Self::I32.as_scalar().expect(\"checked scalar definition\").vector(4, false)")
    );
    assert!(
        output
            .scalars
            .contains("pub const COPY: Self = Self::I32.as_scalar().expect(\"checked scalar definition\").vector(4, false)")
    );
    assert!(
        output
            .scalars
            .contains("pub const MASK: Self = Self::BOOL.as_scalar().expect(\"checked scalar definition\").vector(8, true)")
    );
    let equivalent = source
        .replace("vector(WORD, 4)", "vector(int(32), 4)")
        .replace("type WORD = LATER;", "type WORD = int(32);");
    let equivalent = compile_mir(&equivalent).unwrap();
    assert_eq!(output.scalars, equivalent.scalars);
    assert_eq!(output.builtins, equivalent.builtins);
    let reversed = source.lines().rev().collect::<Vec<_>>().join("\n");
    assert_eq!(output.scalars, compile_mir(&reversed).unwrap().scalars);
}

#[test]
fn aliases_work_in_operation_signatures_and_set_expressions() {
    let output = compile_mir(
        r#"
        type WORD = I32;
        type WIDE = vector(WORD, scalable(4));
        format Pair { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
        op Add<T: WORD | WIDE>(lhs: T, rhs: T) -> T {
            mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
        }
    "#,
    )
    .unwrap();
    assert!(output.builtins.contains("3 => 0x00040001,"));
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

    let output = veloc_opgen::compile_mir(&BUILTINS.replace("I32(3)", "I32(9)")).unwrap();
    assert!(
        output
            .scalars
            .contains("pub const I32: Self = Self(9 << SCALAR_SHIFT);")
    );
    assert!(output.scalars.contains("9 => Some(Self::I32)"));
}

#[test]
fn deeply_nested_type_construction_is_bounded_by_the_parser() {
    let source = format!("type A = {}I32{};", "vector(".repeat(70), ", 4)".repeat(70));
    rejected(&common::source(&source), "nesting exceeds 64");
}
