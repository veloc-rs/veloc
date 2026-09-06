mod common;

use veloc_opgen::generate_lowering;
use veloc_semantics::BvOp;

const OPS: &str = r#"
format Binary { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
op Direct<T: ScalarInteger>(lhs: T, rhs: T) -> T {
    mnemonic: "direct", storage: Binary { args: [lhs, rhs] }, semantics: bv.sub(lhs, rhs)
}
op Reversed<T: ScalarInteger>(lhs: T, rhs: T) -> T {
    mnemonic: "reversed", storage: Binary { args: [lhs, rhs] }, semantics: bv.sub(rhs, lhs)
}
op Composed<T: ScalarInteger>(lhs: T, rhs: T) -> T {
    mnemonic: "composed", storage: Binary { args: [lhs, rhs] }, semantics: bv.add(lhs, bv.sub(rhs, bv.one()))
}
op Trapping<T: ScalarInteger>(lhs: T, rhs: T) -> T {
    mnemonic: "trapping", storage: Binary { args: [lhs, rhs] }, semantics: bv.udiv(lhs, rhs),
    traps: [DivisionByZero(bv.eq(rhs, bv.zero()))]
}
op Multiple<T: ScalarInteger>(lhs: T, rhs: T) -> (T, BOOL) {
    mnemonic: "multiple", storage: Binary { args: [lhs, rhs] }, semantics: [bv.add(lhs, rhs), bv.eq(lhs, rhs)]
}
"#;

#[test]
fn direct_mapping_excludes_traps_reordering_compositions_and_extra_results() {
    let defs = common::parse(OPS).unwrap();
    let code = generate_lowering(
        &defs,
        &[
            (BvOp::Sub, "G_SUB"),
            (BvOp::Add, "G_ADD"),
            (BvOp::UDiv, "G_UDIV"),
        ],
    )
    .unwrap();
    assert!(code.contains("Opcode::Direct => Some(veloc_lir::GenericOpcode::G_SUB)"));
    for name in ["Reversed", "Composed", "Trapping", "Multiple"] {
        assert!(!code.contains(&format!("Opcode::{name} =>")));
    }
    assert!(!code.contains("BvOp"));
    assert!(!code.contains(".spec()"));
    assert!(!code.contains(".iter()"));
}

#[test]
fn conflicting_bindings_are_build_errors_not_first_match_wins() {
    let defs = common::parse(OPS).unwrap();
    for bindings in [
        [(BvOp::Sub, "G_SUB"), (BvOp::Sub, "G_OTHER")],
        [(BvOp::Sub, "G_SUB"), (BvOp::Add, "G_SUB")],
    ] {
        assert!(
            generate_lowering(&defs, &bindings)
                .unwrap_err()
                .message
                .contains("ambiguous")
        );
    }
}
