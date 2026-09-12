mod common;

use common::{BUILTINS, TYPES};

fn rejected(source: &str, message: &str) {
    let result = std::panic::catch_unwind(|| veloc_opgen::compile(source))
        .expect("bad encodings must produce diagnostics, not panic");
    let error = result.err().expect("encoding should be rejected");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn scalar_code_validation_uses_the_declared_field_width() {
    let wide = BUILTINS
        .replace("scalar(4)", "scalar(5)")
        .replace("I8(1)", "I8(31)");
    assert!(
        veloc_opgen::compile(&wide)
            .unwrap()
            .types
            .contains("I8 = 31,")
    );
    rejected(&wide.replace("I8(31)", "I8(32)"), "1..=31");
    rejected(&BUILTINS.replace("scalar(4)", "scalar(3)"), "1..=7");
}

#[test]
fn encodings_are_explicit_and_can_be_forward_referenced() {
    let builtins = include_str!("../../defs/builtins.ops")
        .strip_prefix("import \"types.ops\";\n")
        .unwrap();
    rejected(builtins, "missing encoding Type");
    let first = veloc_opgen::compile(&BUILTINS).unwrap();
    let last = veloc_opgen::compile(&format!("{builtins}\n{TYPES}")).unwrap();
    assert_eq!(first.types, last.types);
    rejected(&format!("{}\n{TYPES}", *BUILTINS), "duplicate");
    rejected(
        &TYPES.replace("encoding Type", "encoding Other"),
        "missing encoding Type",
    );
}

#[test]
fn malformed_or_unsupported_layouts_are_rejected_before_generation() {
    for (from, to, error) in [
        ("storage: u16", "storage: u32", "require u16 storage"),
        ("storage: u16,", "", "missing `storage`"),
        ("scalar(4)", "scalar(0)", "positive widths"),
        ("scalar(4)", "scalar(4294967295)", "fit u16"),
        ("scalar(4)", "scalar(16)", "fit u16"),
        ("scalable(1)", "scalable(16)", "fit u16"),
        ("scalar(4)", "scalar(9)", "MIR API representation"),
        ("lanes_log2(4)", "lanes_log2(5)", "MIR API representation"),
        ("scalable(1)", "scalable(2)", "MIR API representation"),
        ("scalar(4)", "unknown(4)", "unknown Type encoding field"),
        ("scalar(4)", "scalar(4), scalar(4)", "duplicate bit field"),
        ("scalar(4)", "scalar", "expected encoding field(bits)"),
        ("scalar(4)", "scalar(4, 4)", "expected encoding field(bits)"),
        ("scalar(4)", "scalar(width)", "width must be a number"),
        ("scalar(4), ", "", "missing field `scalar`"),
        ("lanes_log2(4), ", "", "missing field `lanes_log2`"),
        (", scalable(1)", "", "missing field `scalable`"),
        (
            "storage: u16,",
            "storage: u16, typo: 1,",
            "unknown field `typo`",
        ),
    ] {
        rejected(&TYPES.replace(from, to), error);
    }
}
