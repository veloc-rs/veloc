mod common;

use common::{BUILTINS, TYPES};

fn rejected(source: &str, message: &str) {
    let result = std::panic::catch_unwind(|| veloc_opgen::compile_mir(source))
        .expect("bad encodings must produce diagnostics, not panic");
    let error = result.err().expect("encoding should be rejected");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn layout_generates_masks_shifts_limits_and_the_type_storage() {
    let output = veloc_opgen::compile_mir(BUILTINS).unwrap().encoding;
    for expected in [
        "pub struct Type(u16)",
        "const SCALAR_MASK: u16 = 0x000f;",
        "const SCALAR_SHIFT: u32 = 0;",
        "const LANES_LOG2_MASK: u16 = 0x00f0;",
        "const LANES_LOG2_SHIFT: u32 = 4;",
        "const LANES_LOG2_MAX: u16 = 15;",
        "const SCALABLE_MASK: u16 = 0x0100;",
        "const USED_MASK: u16 = 0x01ff;",
        "const fn element_code(self) -> u8",
        "const fn lanes_log2(self) -> u16",
        "pub const fn is_scalable(self) -> bool",
        "pub const fn element_type(self) -> Self",
        "pub const fn to_raw(self) -> u16",
    ] {
        assert!(output.contains(expected), "{expected}");
    }
}

#[test]
fn field_order_and_widths_determine_the_encoding() {
    let source = BUILTINS.replace(
        "scalar(4), lanes_log2(4), scalable(1)",
        "scalable(1), scalar(5), lanes_log2(3)",
    );
    let output = veloc_opgen::compile_mir(&source).unwrap().encoding;
    for expected in [
        "const SCALABLE_MASK: u16 = 0x0001;",
        "const SCALAR_MASK: u16 = 0x003e;",
        "const SCALAR_SHIFT: u32 = 1;",
        "const LANES_LOG2_MASK: u16 = 0x01c0;",
        "const LANES_LOG2_SHIFT: u32 = 6;",
        "const LANES_LOG2_MAX: u16 = 7;",
        "const USED_MASK: u16 = 0x01ff;",
    ] {
        assert!(output.contains(expected), "{expected}");
    }
    let narrower = BUILTINS.replace("lanes_log2(4)", "lanes_log2(3)");
    assert!(
        veloc_opgen::compile_mir(&narrower)
            .unwrap()
            .encoding
            .contains("const USED_MASK: u16 = 0x00ff;")
    );
}

#[test]
fn scalar_code_validation_uses_the_declared_field_width() {
    let wide = BUILTINS
        .replace("scalar(4)", "scalar(5)")
        .replace("I8(1)", "I8(31)");
    assert!(
        veloc_opgen::compile_mir(&wide)
            .unwrap()
            .scalars
            .contains("pub const I8: Self = Self(31 << SCALAR_SHIFT);")
    );
    rejected(&wide.replace("I8(31)", "I8(32)"), "1..=31");
    rejected(&BUILTINS.replace("scalar(4)", "scalar(3)"), "1..=7");
}

#[test]
fn encodings_are_explicit_and_can_be_forward_referenced() {
    let builtins = include_str!("../../mir/defs/builtins.ops");
    rejected(builtins, "missing encoding Type");
    let first = veloc_opgen::compile_mir(BUILTINS).unwrap();
    let last = veloc_opgen::compile_mir(&format!("{builtins}\n{TYPES}")).unwrap();
    assert_eq!(first.encoding, last.encoding);
    assert_eq!(first.scalars, last.scalars);
    rejected(&format!("{BUILTINS}\n{TYPES}"), "duplicate encoding");
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
