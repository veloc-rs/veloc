use veloc_opgen::compile_mir;

const FORMATS: &str = include_str!("../../ir/defs/formats.ops");

fn definitions() -> String {
    [FORMATS, include_str!("../../ir/defs/mir.ops")].join("\n")
}

fn changed_record(kind: &str, name: &str, from: &str, to: &str) -> String {
    let source = definitions();
    let start = source.find(&format!("{kind} {name} {{")).unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let record = &source[start..end];
    assert!(record.contains(from), "{kind} {name} has no `{from}`");
    source.replacen(record, &record.replacen(from, to, 1), 1)
}

fn rejected(source: &str, expected: &str) {
    let error = match compile_mir(source) {
        Ok(_) => panic!("invalid storage contract was accepted"),
        Err(error) => error,
    };
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn every_specialized_codec_checks_all_field_names() {
    for record in FORMATS.split("\nformat ").skip(1) {
        let body = record.split("\n}").next().unwrap();
        if body.contains("text: values(") || body.contains("text: nullary") {
            continue;
        }
        let layout = body.split_whitespace().next().unwrap();
        let fields = body
            .split("fields: [")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        for field in fields.split(", ") {
            let name = field.split('(').next().unwrap();
            let source = changed_record(
                "format",
                layout,
                &format!("{name}("),
                &format!("wrong_{name}("),
            );
            rejected(&source, "field contract");
        }
    }
}

#[test]
fn codec_contracts_check_property_types_and_operand_order() {
    for (layout, from, to) in [
        ("Iconst", "value(u64)", "value(u32)"),
        ("Bconst", "value(bool)", "value(u64)"),
        ("Load", "offset(u32)", "offset(i32)"),
        (
            "Store",
            "ptr(Value), value(Value)",
            "value(Value), ptr(Value)",
        ),
        ("IntCompare", "kind(IntCC)", "kind(FloatCC)"),
        ("VectorGather", "ext(VectorMemExtId)", "ext(VectorExtId)"),
    ] {
        rejected(
            &changed_record("format", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn codec_contracts_distinguish_arrays_from_fixed_and_variadic_lists() {
    for (layout, from, to) in [
        ("IntCompare", "args(values(2))", "args(list(2))"),
        ("VectorStoreStrided", "args(list(3))", "args(values(3))"),
        ("VectorScatter", "args(list(3))", "args(ValueList)"),
        ("Shuffle", "args(values(2))", "args(values(3))"),
    ] {
        rejected(
            &changed_record("format", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn codec_contracts_reject_missing_and_unprinted_properties() {
    for (layout, from, to) in [
        ("Iconst", "fields: [value(u64)]", "fields: []"),
        ("Iconst", "value(u64)", "value(u64), unused(u32)"),
        ("Load", ", flags(MemFlags)", ""),
        ("CallIndirect", ", sig_id(SigId)", ""),
    ] {
        rejected(
            &changed_record("format", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn specialized_codecs_reject_unadapted_alternate_layouts() {
    for fields in ["[ptr(Value)]", "[ptr(Value), offset(u32), flags(MemFlags)]"] {
        let source = format!(
            "{}\nlayout Other {{ fields: {fields}, opcode: fixed(Load), format: fixed(Load) }}",
            definitions()
        );
        rejected(&source, "no text adapter");
    }
}

#[test]
fn generic_alternates_cannot_silently_drop_auxiliary_state() {
    for fields in [
        "[opcode(Opcode), args(ValueList), ext(VectorExtId)]",
        "[opcode(Opcode), args(ValueList), hidden(u32)]",
    ] {
        let source = format!(
            "{}\nlayout Other {{ fields: {fields}, opcode: dynamic(opcode), format: arity(args, [Unary, Binary, Ternary]) }}",
            definitions()
        );
        rejected(&source, "no text adapter");
    }
}

#[test]
fn existing_predication_has_a_checked_supported_adapter() {
    assert!(compile_mir(&definitions()).is_ok());
    for (from, to) in [
        ("ext(VectorExtId)", "config(VectorExtId)"),
        ("ext(VectorExtId)", "ext(VectorMemExtId)"),
        ("ext(VectorExtId)", "ext(VectorExtId), hidden(u32)"),
    ] {
        rejected(
            &changed_record("layout", "VectorOpWithExt", from, to),
            "field contract",
        );
    }
}

#[test]
fn canonical_dynamic_layouts_preserve_their_public_field_contracts() {
    for (layout, from, to) in [
        ("Unary", "arg(Value)", "operand(Value)"),
        ("Binary", "args(values(2))", "inputs(values(2))"),
        ("Ternary", "args(values(3))", "inputs(values(3))"),
        ("IntToPtr", "arg(Value)", "operand(Value)"),
    ] {
        rejected(
            &changed_record("format", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn custom_value_formats_allow_custom_field_names() {
    let source = r#"
        format Pair {
            fields: [op(Opcode), left(Value), right(Value)],
            opcode: dynamic(op), text: values(2)
        }
        op Add<T: Integer>(left: T, right: T) -> (result: T) {
            mnemonic: "add", storage: Pair { left: left, right: right },
            traits: [], memory: NONE
        }
    "#;
    assert!(compile_mir(source).is_ok());
}
