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
        assert_eq!(actual.type_rules, expected.type_rules, "{expression}");
        assert_eq!(actual.opcodes, expected.opcodes, "{expression}");
    }
}

#[test]
fn named_aliases_and_inline_expressions_intern_to_the_same_set() {
    let source = format!("class Wide {{ members: [I32 | I64] }}\n{}", pair("Wide"));
    let named = compile_mir(&source).unwrap();
    let inline = compile_mir(&source.replace("T: Wide", "T: I64 | I32")).unwrap();
    assert_eq!(named.type_rules, inline.type_rules);
    assert_eq!(named.opcodes, inline.opcodes);
    assert!(inline.type_rules.contains("C::Wide.accepts(operands[0])"));
    assert!(inline.type_rules.contains("operands[1] == operands[0]"));

    let aliases =
        compile_mir("class A { members: [I32 | I64] } class B { members: [I64 | I32] }").unwrap();
    let id = |name| {
        aliases
            .opcodes
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
fn expression_nesting_is_bounded_but_flat_unions_are_not_recursive() {
    let deep = format!("{}I32{}", "(".repeat(70), ")".repeat(70));
    rejected(&pair(&deep), "nesting exceeds 64");
    let flat = vec!["I32"; 1000].join(" | ");
    assert!(compile_mir(&pair(&flat)).is_ok());
}
