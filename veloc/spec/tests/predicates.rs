mod common;

use common::BUILTINS;

#[test]
fn public_predicate_meanings_come_from_defs_not_rust_name_allowlists() {
    let changed = BUILTINS.replace(
        "predicate is_integer = Integer;",
        "predicate is_integer = I32 | I64;",
    );
    let output = veloc_opgen::compile_mir(&changed).unwrap();
    let method = output
        .scalars
        .rsplit("pub const fn is_integer(self)")
        .next()
        .unwrap()
        .split("/// Membership")
        .next()
        .unwrap();
    assert!(method.contains("3 | 4 => 0x00000001,"));
    assert!(!method.contains("self.element_type().is_integer()"));
}
