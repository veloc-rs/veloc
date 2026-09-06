mod common;

use veloc_opgen::generate_lowering;
use veloc_semantics::BvOp;

#[test]
fn conflicting_bindings_are_build_errors_not_first_match_wins() {
    let defs = common::parse("").unwrap();
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
