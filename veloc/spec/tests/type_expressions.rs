mod common;

use common::compile_mir;

const PAIR: &str = r#"
format Pair { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
op Add<T: DOMAIN>(lhs: T, rhs: T) -> T {
    mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
}
"#;

fn pair(domain: &str) -> String {
    PAIR.replace("DOMAIN", domain)
}

fn rejected(source: &str, expected: &str) {
    let error = std::panic::catch_unwind(|| compile_mir(source))
        .expect("invalid definitions must not panic")
        .err()
        .expect("invalid definition was accepted");
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn precedence_parentheses_and_equivalent_sets_share_runtime_checks() {
    for (expression, equivalent) in [
        ("I32 | I64 & ScalarFloat", "I32"),
        ("(I32 | I64) & ScalarInteger", "I32 | I64"),
        (
            "(Integer | Float) & Vector",
            "vectors(ScalarInteger | ScalarFloat)",
        ),
        ("Integer & Vector", "vectors(ScalarInteger)"),
        ("I32 | (I64 & F64)", "I32"),
        ("I32 | I64 | I32", "I64 | I32"),
        ("I32 | vectors(I64 & F64)", "I32"),
        (
            "Integer | BOOL | vectors(BOOL)",
            "BOOL | vectors(BOOL) | Integer",
        ),
    ] {
        // Some sets include floats, so this test is about constraints, not BV semantics.
        let source = |expr| pair(expr).replace("semantics: bv.add(lhs, rhs)", "memory: NONE");
        let actual = compile_mir(&source(expression)).unwrap();
        let expected = compile_mir(&source(equivalent)).unwrap();
        assert_eq!(actual.types, expected.types, "{expression}");
        assert_eq!(actual.builtins, expected.builtins, "{expression}");
    }
}

#[test]
fn named_aliases_and_inline_expressions_intern_to_the_same_set() {
    let source = format!("class Wide {{ members: [I32 | I64] }}\n{}", pair("Wide"));
    let named = compile_mir(&source).unwrap();
    let inline = compile_mir(&source.replace("T: Wide", "T: I64 | I32")).unwrap();
    assert_eq!(named.types, inline.types);
    assert_eq!(named.builtins, inline.builtins);
    assert!(inline.types.contains("P::Bind(0, C::Wide), P::Same(0)"));
    assert!(inline.types.contains("P::Same(0)"));

    let aliases =
        compile_mir("class A { members: [I32 | I64] } class B { members: [I64 | I32] }").unwrap();
    let id = |name| {
        aliases
            .builtins
            .lines()
            .find(|line| line.starts_with(&format!("pub const {name}:")))
            .unwrap()
            .split("= ")
            .nth(1)
            .unwrap()
    };
    assert_eq!(id("A"), id("B"));
}

#[test]
fn direct_patterns_and_shape_constraints_accept_expressions() {
    let source = r#"
format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
op Copy(arg: I32 | I64) -> I32 | I64 {
    mnemonic: "copy", storage: Unary { arg: arg }, memory: NONE
}
op Lane<T: vectors(I32 | I64)>(arg: T) -> element(T) {
    mnemonic: "lane", storage: Unary { arg: arg }, memory: NONE
}
op Convert<T: I32 | I64>(arg: T) -> shape(T, Float & Scalar) {
    mnemonic: "convert", storage: Unary { arg: arg }, memory: NONE
}
"#;
    let output = compile_mir(source).unwrap();
    assert!(output.types.contains("P::Class(C("));
    assert!(output.types.contains("P::ShapeOf(0, C::ScalarFloat)"));
    assert!(
        output
            .opcodes
            .contains("pub fn lane(&mut self, arg: crate::Value) -> crate::Value")
    );
    rejected(
        &source.replace("shape(T, Float & Scalar)", "shape(T, Float & Vector)"),
        "impossible shape",
    );
    rejected(
        &source.replace("vectors(I32 | I64)", "vectors(T)"),
        "unknown type or class `T`",
    );
}

#[test]
fn semantics_and_text_codecs_use_resolved_inline_constraints() {
    for domain in ["I32 | I64", "Integer & Vector"] {
        assert!(compile_mir(&pair(domain)).is_ok(), "{domain}");
    }
    assert!(compile_mir(&pair("BOOL | vectors(BOOL)").replace("bv.add", "bv.and")).is_ok());
    rejected(&pair("BOOL | vectors(BOOL)"), "expected a bitvector");
    rejected(&pair("I32 | F64"), "floating-point");
    let source = r#"
format Literal { fields: [opcode(Opcode), value(u64)], opcode: dynamic(opcode) }
op Literal<T: Float & Scalar>(@value: u64) -> T {
    mnemonic: "literal", storage: Literal { value: value },
    text: Text { args: [float(value)] }, memory: NONE
}
"#;
    assert!(compile_mir(source).is_ok());
    rejected(
        &source.replace("Float & Scalar", "Float & Vector"),
        "scalar float",
    );
}

#[test]
fn invalid_expressions_empty_constraints_and_nested_cycles_are_diagnosed() {
    for (domain, message) in [
        ("Integer & Float", "must not be empty"),
        ("vectors(I32) & I32", "must not be empty"),
        ("Integer | Missing", "unknown type or class `Missing`"),
        ("(I32 & F32) & Missing", "unknown type or class `Missing`"),
        ("vectors(I32 | PTR)", "non-pointer scalar"),
        ("Integer |", "expected a name"),
        ("Integer || BOOL", "expected a name"),
        ("Integer && Vector", "expected a name"),
        ("Integer + BOOL", "expected `,`"),
        ("(Integer | BOOL", "expected `)`"),
        ("()", "expected a name"),
    ] {
        rejected(&pair(domain), message);
    }
    rejected(
        "class Empty { members: [Integer & Float] }",
        "must not be empty",
    );
    rejected(
        "class A { members: [I32 | vectors(B)] } class B { members: [A & Scalar] }",
        "cyclic type class",
    );
    assert!(
        compile_mir("class A { members: [vectors(B & Scalar)] } class B { members: [I32 | I64] }")
            .is_ok()
    );
}

#[test]
fn expression_nesting_is_bounded_but_flat_unions_are_not_recursive() {
    let deep = format!("{}I32{}", "(".repeat(70), ")".repeat(70));
    rejected(&pair(&deep), "nesting exceeds 64");
    let flat = vec!["I32"; 1000].join(" | ");
    assert!(compile_mir(&pair(&flat)).is_ok());
}
