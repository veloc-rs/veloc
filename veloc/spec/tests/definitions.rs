mod common;
use common::compile_mir;

#[test]
fn the_actual_mir_definitions_compile_deterministically() {
    let source = [
        include_str!("../../mir/defs/formats.ops"),
        include_str!("../../mir/defs/mir.ops"),
    ]
    .join("\n");
    let first = compile_mir(&source).unwrap();
    let second = compile_mir(&source).unwrap();
    assert_eq!(first.types, second.types);
    assert_eq!(first.builders, second.builders);
    assert_eq!(first.type_rules, second.type_rules);
    assert_eq!(first.validation, second.validation);
    assert_eq!(first.opcodes, second.opcodes);
    assert_eq!(first.instructions, second.instructions);
    assert_eq!(first.text_parser, second.text_parser);
    assert_eq!(first.text_printer, second.text_printer);
}
