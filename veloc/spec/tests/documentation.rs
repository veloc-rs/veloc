#[test]
fn checked_definitions_generate_reference_without_runtime_reflection() {
    let source = concat!(
        include_str!("../../mir/defs/types.ops"),
        "\n",
        include_str!("../../mir/defs/builtins.ops"),
        "\n",
        include_str!("../../mir/defs/comparisons.ops"),
        "\n",
        include_str!("../../mir/defs/formats.ops"),
        "\n",
        include_str!("../../mir/defs/mir.ops"),
    );
    let definitions = veloc_opgen::parse(source).unwrap();
    let output = veloc_opgen::compile_mir(source).unwrap();
    assert_eq!(
        output
            .documentation
            .lines()
            .filter(|line| line.starts_with("| `"))
            .count(),
        definitions.operation_count()
    );
    assert!(output.documentation.contains("| `iadd` |"));
    assert!(
        output
            .documentation
            .contains("wider(operands[0], results[0])")
    );
    assert!(
        output
            .documentation
            .contains("pointer comparison only supports eq and ne")
    );
    assert!(!output.opcodes.contains("type_scheme:"));
    assert!(!output.opcodes.contains("constraints:"));
}
