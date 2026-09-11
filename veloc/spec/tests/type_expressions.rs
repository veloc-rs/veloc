mod common;

use common::compile;

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
    let error = std::panic::catch_unwind(|| compile(source))
        .expect("invalid definitions must not panic")
        .err()
        .expect("invalid definition was accepted");
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn expression_nesting_is_bounded_but_flat_unions_are_not_recursive() {
    let deep = format!("{}I32{}", "(".repeat(70), ")".repeat(70));
    rejected(&pair(&deep), "nesting exceeds 64");
    let flat = vec!["I32"; 1000].join(" | ");
    assert!(compile(&pair(&flat)).is_ok());
}
