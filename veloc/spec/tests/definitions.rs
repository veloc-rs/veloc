mod common;

#[test]
fn the_actual_mir_definitions_compile_deterministically() {
    let source = [
        include_str!("../../mir/defs/formats.ops"),
        include_str!("../../mir/defs/mir.ops"),
    ]
    .join("\n");
    let source = common::source(&source);
    let plan = veloc_opgen::plan(&source).unwrap();
    assert!(plan.definitions().operation_count() > 0);
    let first = plan.generate();
    let second = plan.generate();
    assert_eq!(
        first.instructions,
        veloc_opgen::compile(&source).unwrap().instructions
    );
    assert_eq!(first.types, second.types);
    assert_eq!(first.builders, second.builders);
    assert_eq!(first.type_rules, second.type_rules);
    assert_eq!(first.validation, second.validation);
    assert_eq!(first.opcodes, second.opcodes);
    assert_eq!(first.instructions, second.instructions);
    assert_eq!(first.text_parser, second.text_parser);
    assert_eq!(first.text_printer, second.text_printer);
    assert_eq!(first.evaluation, second.evaluation);
    assert_eq!(first.semantics, second.semantics);
}

#[test]
fn output_errors_are_rejected_before_emission() {
    let base = r#"
struct Custom { arg: Value }
op Example<T: Integer>(arg: T) -> T {
    meta: OpInfo { memory: Known([]) },
    mnemonic: "example", storage: Custom { arg: arg },
}
"#;
    for (source, message) in [
        ("storage Operands {} property Int { constraints: [require(true, \"valid\")] }".into(), "property validators"),
        (base.replace("mnemonic: \"example\"", "mnemonic: \"emit\""), "InstBuilder method"),
        (base.replace("storage: Custom { arg: arg },", "storage: Custom { arg: arg }, text: \"{missing}\","), "missing"),
        (format!("{base}\nstruct Alternate {{ arg: Value, extra: u32 }}\nlayout Alternate {{ format: fixed(Custom), text: \"{{arg}}, extra={{extra}}\", constraints: [unknown > 0] }}"), "unknown constraint name"),
        ("storage Operands {} struct Unary { dst: Def, src: Use } op G_COPY<T: Integer>(src: T) -> T { meta: OpInfo { memory: Known([]) }, storage: Unary, constraints: [require(true, \"ok\")] }".into(), "operand storage does not yet support"),
    ] {
        let source = common::source(&source);
        veloc_opgen::parse(&source).unwrap();
        let error = veloc_opgen::plan(&source).err().expect("invalid output plan");
        assert!(error.message.contains(message), "{error}");
    }
}

#[test]
fn property_constraints_are_checked_without_an_output_plan() {
    let source = common::source("property Int { constraints: [unknown > 0] }");
    let error = veloc_opgen::parse(&source)
        .err()
        .expect("invalid property contract");
    assert!(error.message.contains("unknown constraint name"), "{error}");
}
