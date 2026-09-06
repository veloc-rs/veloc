mod common;
use common::compile_mir;

const FORMATS: &str = include_str!("../../mir/defs/formats.ops");

fn definitions() -> String {
    [FORMATS, include_str!("../../mir/defs/mir.ops")].join("\n")
}

fn changed_record(kind: &str, name: &str, from: &str, to: &str) -> String {
    let source = definitions();
    let prefix = if kind == "op" {
        format!("op {name}")
    } else {
        format!("{kind} {name} {{")
    };
    let start = source.find(&prefix).unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let record = &source[start..end];
    assert!(record.contains(from), "{kind} {name} has no `{from}`");
    source.replacen(record, &record.replacen(from, to, 1), 1)
}

fn rejected(source: &str, expected: &str) {
    let error = match compile_mir(source) {
        Ok(_) => panic!("invalid definition was accepted:\n{source}"),
        Err(error) => error,
    };
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn existing_runtime_layouts_keep_their_public_field_names() {
    for record in FORMATS.split("\nformat ").skip(1) {
        let body = record.split("\n}").next().unwrap();
        let layout = body.split_whitespace().next().unwrap();
        let fields = body
            .split("fields: [")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        for field in fields.split(", ") {
            if field.is_empty() {
                continue;
            }
            let name = field.split('(').next().unwrap();
            let source = changed_record(
                "format",
                layout,
                &format!("{name}("),
                &format!("wrong_{name}("),
            );
            let expected = if name == "opcode" {
                "unknown storage field `opcode`"
            } else {
                "field contract"
            };
            rejected(&source, expected);
        }
    }
}

#[test]
fn runtime_layout_contracts_check_property_types_and_operand_order() {
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
fn runtime_layout_contracts_distinguish_arrays_from_fixed_and_variadic_lists() {
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
fn runtime_layout_contracts_reject_missing_and_extra_properties() {
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
fn text_projections_cover_every_logical_parameter_once() {
    for args in ["[lhs]", "[lhs, lhs]", "[lhs, missing]"] {
        rejected(
            &changed_record(
                "op",
                "IAdd",
                "storage: Binary { args: [lhs, rhs] },",
                &format!("storage: Binary {{ args: [lhs, rhs] }}, text: Text {{ args: {args} }},"),
            ),
            "",
        );
    }
    rejected(
        &changed_record("op", "Load", "args: [ptr]", "args: [ptr, offset]"),
        "offset",
    );
}

#[test]
fn typed_text_atoms_do_not_accept_incompatible_fields() {
    for atom in ["integer(ptr)", "float(ptr)", "bytes(ptr)"] {
        rejected(
            &changed_record("op", "Load", "args: [ptr]", &format!("args: [{atom}]")),
            "",
        );
    }
    for named in [
        "optional(offset)",
        "default(offset, -1)",
        "default(offset, true)",
    ] {
        rejected(
            &changed_record("op", "Load", "default(offset, 0)", named),
            "",
        );
    }
}

#[test]
fn compound_property_paths_and_optional_values_are_checked() {
    for (from, to) in [
        ("optional(mem.mask)", "optional(mem.unknown)"),
        ("optional(mem.evl)", "optional(mem.offset)"),
        ("flags: mem.flags", "flags: mem.offset"),
        (
            "default(mem.offset, 0)",
            "default(mem.offset, 0), optional(mem.mask)",
        ),
    ] {
        rejected(&changed_record("op", "Gather", from, to), "");
    }
}

#[test]
fn signature_results_require_a_typed_signature_source() {
    for (op, from, to) in [
        ("Call", "signature: function(func_id),", ""),
        ("CallIndirect", "signature: sig_id,", ""),
        ("CallIndirect", "signature: sig_id", "signature: ptr"),
        ("Call", "signature: function(func_id)", "signature: func_id"),
    ] {
        rejected(&changed_record("op", op, from, to), "signature");
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
            opcode: dynamic(op)
        }
        op Add<T: Integer>(left: T, right: T) -> (result: T) {
            mnemonic: "add", storage: Pair { left: left, right: right },
            traits: [], memory: NONE
        }
    "#;
    assert!(compile_mir(source).is_ok());
}

#[test]
fn format_level_codec_names_are_not_a_second_text_definition() {
    rejected(
        &changed_record(
            "format",
            "Iconst",
            "opcode: fixed(Iconst)",
            "opcode: fixed(Iconst), text: IntegerConstant",
        ),
        "text",
    );
}
