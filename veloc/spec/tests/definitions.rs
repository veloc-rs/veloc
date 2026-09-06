use veloc_opgen::{compile_mir, parse};

const ADD: &str = r#"
format Binary {
    fields: [opcode(Opcode), args(values(2))],
    opcode: dynamic(opcode)
}
op IAdd<T: Integer>(lhs: T, rhs: T) -> (result: T) {
    mnemonic: "iadd", storage: Binary { args: [lhs, rhs] },
    memory: NONE, semantics: bv.add(lhs, rhs)
}
"#;

fn rejected(source: &str, expected: &str) {
    let error = match compile_mir(source) {
        Ok(_) => panic!("definition unexpectedly accepted"),
        Err(error) => error,
    };
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn one_definition_drives_all_mir_views() {
    let definitions = parse(ADD).unwrap();
    assert_eq!(definitions.operation_count(), 1);
    assert_eq!(definitions.format_count(), 1);
    let output = compile_mir(ADD).unwrap();
    assert!(output.opcodes.contains("pub enum Opcode"));
    assert!(output.opcodes.contains("pub fn iadd("));
    assert!(output.types.contains("P::Bind(0, C::Integer)"));
    assert!(output.instructions.contains("pub enum InstructionData"));
    assert!(output.instructions.contains("visit_operands"));
    assert!(output.instructions.contains("replace_value"));
    assert!(output.instructions.contains("from_values"));
    assert!(output.text_parser.contains("Opcode::IAdd"));
    assert!(output.text_printer.contains("Opcode::IAdd"));
}

#[test]
fn the_actual_mir_definitions_compile_deterministically() {
    let source = [
        include_str!("../../mir/defs/formats.ops"),
        include_str!("../../mir/defs/mir.ops"),
    ]
    .join("\n");
    let first = compile_mir(&source).unwrap();
    let second = compile_mir(&source).unwrap();
    assert_eq!(first.formats, second.formats);
    assert_eq!(first.types, second.types);
    assert_eq!(first.opcodes, second.opcodes);
    assert_eq!(first.instructions, second.instructions);
    assert_eq!(first.text_parser, second.text_parser);
    assert_eq!(first.text_printer, second.text_printer);
}

#[test]
fn unknown_and_duplicate_fields_are_diagnostics() {
    rejected(
        &ADD.replace("memory: NONE", "memory: NONE, memroy: NONE"),
        "unknown field",
    );
    rejected(
        &ADD.replace("memory: NONE", "memory: NONE, memory: NONE"),
        "duplicate field",
    );
}

#[test]
fn names_and_references_are_checked_before_rust_generation() {
    rejected(
        &ADD.replace("storage: Binary", "storage: Missing"),
        "unknown",
    );
    rejected(
        &ADD.replace("\"iadd\"", "\"type\""),
        "invalid generated identifier",
    );
    rejected(
        &ADD.replace("rhs: T", "rhs: U"),
        "unbound type variable `U`",
    );
    let op = ADD.split("op IAdd").nth(1).unwrap();
    rejected(&format!("{ADD}\nop Other{op}"), "duplicate mnemonic");
}

#[test]
fn logical_parameter_names_drive_builders_storage_and_semantics() {
    let source = ADD.replace("lhs", "left").replace("rhs", "right");
    let output = compile_mir(&source).unwrap();
    assert!(output.opcodes.contains(
        "pub fn iadd(&mut self, left: crate::Value, right: crate::Value) -> crate::Value"
    ));
    assert!(output.opcodes.contains("args: [left, right]"));

    rejected(
        &source.replace("args: [left, right]", "args: [left, missing]"),
        "missing",
    );
    rejected(
        &source.replace("bv.add(left, right)", "bv.add(left, missing)"),
        "missing",
    );
}

#[test]
fn removed_type_scheme_configuration_is_rejected() {
    rejected(
        &ADD.replace("memory: NONE", "types: INTEGER_BINARY, memory: NONE"),
        "types",
    );
    rejected(
        "types INTEGER_BINARY { operands: [], results: [] }",
        "types",
    );
}

#[test]
fn incompatible_layouts_and_semantics_are_rejected() {
    rejected(
        &ADD.replace("args: [lhs, rhs]", "args: [lhs]"),
        "storage field `args` requires 2 arguments",
    );
    rejected(&ADD.replace("bv.add", "bv.unknown"), "unknown");
    rejected(&ADD.replace("T: Integer", "T: Float"), "integer");
    rejected(
        &ADD.replace("bv.add", "bv.neg"),
        "bv.neg expects 1 arguments, got 2",
    );
    rejected(&ADD.replace("memory: NONE", "memory: UNKNOWN"), "pure");
}

#[test]
fn invalid_relations_are_definition_errors() {
    rejected(
        &ADD.replace(
            "storage: Binary",
            "where: [wider(missing, result)], storage: Binary",
        ),
        "missing",
    );
    rejected(
        &ADD.replace(
            "storage: Binary",
            "where: [unknown(lhs, result)], storage: Binary",
        ),
        "type relation",
    );
}

#[test]
fn malformed_syntax_has_a_source_location() {
    rejected(&ADD.replace("memory: NONE", "memory NONE"), "expected `:`");
    rejected(&ADD.replace("\"iadd\"", "\"iadd"), "newline in string");
    rejected(
        &ADD.replace("values(2)", "values(99999999999999)"),
        "out of range",
    );
}

#[test]
fn removed_builder_configuration_is_rejected() {
    for value in ["iadd", "iadd(args)"] {
        rejected(
            &ADD.replace("memory: NONE", &format!("memory: NONE, builder: {value}")),
            "unknown field `builder`",
        );
    }
}
