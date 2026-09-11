mod common;
use common::compile;

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
    let start = source
        .match_indices(&prefix)
        .find_map(|(start, _)| {
            (kind != "op"
                || matches!(
                    source.as_bytes().get(start + prefix.len()),
                    Some(b'(' | b'<')
                ))
            .then_some(start)
        })
        .unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let record = &source[start..end];
    assert!(record.contains(from), "{kind} {name} has no `{from}`");
    source.replacen(record, &record.replacen(from, to, 1), 1)
}

fn rejected(source: &str, expected: &str) {
    let error = match compile(source) {
        Ok(_) => panic!("invalid definition was accepted:\n{source}"),
        Err(error) => error,
    };
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

#[test]
fn runtime_layout_contracts_check_property_types_and_operand_order() {
    for (layout, from, to) in [
        ("Iconst", "value: Int", "value: u32"),
        ("Bconst", "value: bool", "value: u64"),
        ("Load", "offset: u32", "offset: i32"),
        (
            "Store",
            "ptr: Value,\n    value: Value",
            "value: Value,\n    ptr: Value",
        ),
        ("IntCompare", "kind: IntCC", "kind: FloatCC"),
        (
            "VectorGather",
            "ext: VectorMemOptions",
            "ext: VectorExtData",
        ),
    ] {
        rejected(
            &changed_record("record", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn runtime_layout_contracts_check_fixed_and_variadic_groups() {
    for (layout, from, to) in [
        ("IntCompare", "args: values(2)", "args: values(3)"),
        ("VectorStoreStrided", "args: values(3)", "args: values(2)"),
        ("VectorScatter", "args: values(3)", "args: ValueList"),
        ("Shuffle", "args: values(2)", "args: values(3)"),
    ] {
        rejected(
            &changed_record("record", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn runtime_layout_contracts_reject_missing_and_extra_properties() {
    for (layout, from, to) in [
        ("Iconst", "value: Int,", ""),
        ("Iconst", "value: Int", "value: Int, unused: u32"),
        ("Load", "    flags: MemFlags,\n", ""),
        ("CallIndirect", "    sig_id: SigId,\n", ""),
    ] {
        rejected(
            &changed_record("record", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn text_projections_cover_every_logical_parameter_once() {
    for args in ["{lhs}", "{lhs}, {lhs}", "{lhs}, {missing}"] {
        rejected(
            &changed_record(
                "op",
                "IAdd",
                "storage: Binary { args: [lhs, rhs] },",
                &format!("storage: Binary {{ args: [lhs, rhs] }}, text: \"{args}\","),
            ),
            "",
        );
    }
    rejected(
        &changed_record("op", "Load", "{ptr}", "{ptr}, {offset}"),
        "offset",
    );
}

#[test]
fn typed_text_atoms_do_not_accept_incompatible_fields() {
    for atom in ["{ptr:integer}", "{ptr:float}", "{ptr:bytes}"] {
        rejected(&changed_record("op", "Load", "{ptr}", atom), "");
    }
    for named in [
        "[offset={offset}]",
        "offset={offset=-1}",
        "offset={offset=true}",
    ] {
        rejected(&changed_record("op", "Load", "offset={offset}", named), "");
    }
}

#[test]
fn compound_property_paths_and_optional_values_are_checked() {
    for (from, to) in [
        ("{mem.mask}", "{mem.unknown}"),
        ("{mem.evl}", "{mem.offset}"),
        ("{.mem.flags}", "{.mem.offset}"),
        (
            "offset={mem.offset}",
            "offset={mem.offset}[, mask={mem.mask}]",
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
        ("CallValue", "signature: callable(callee),", ""),
        (
            "CallValue",
            "signature: callable(callee)",
            "signature: callable(args)",
        ),
        (
            "CallValue",
            "signature: callable(callee)",
            "signature: callee",
        ),
    ] {
        rejected(&changed_record("op", op, from, to), "signature");
    }
}

#[test]
fn existing_predication_has_a_checked_supported_adapter() {
    assert!(compile(&definitions()).is_ok());
    for (from, to) in [
        ("ext: VectorExtData", "config: VectorExtData"),
        ("ext: VectorExtData", "ext: VectorMemOptions"),
        ("ext: VectorExtData", "ext: VectorExtData, hidden: u32"),
    ] {
        rejected(
            &changed_record("record", "VectorOpWithExt", from, to),
            "field contract",
        );
    }
}

#[test]
fn canonical_dynamic_layouts_preserve_their_public_field_contracts() {
    for (layout, from, to) in [
        ("Unary", "arg: Value", "operand: Value"),
        ("Binary", "args: values(2)", "inputs: values(2)"),
        ("Ternary", "args: values(3)", "inputs: values(3)"),
        ("IntToPtr", "arg: Value", "operand: Value"),
    ] {
        rejected(
            &changed_record("record", layout, from, to),
            "field contract",
        );
    }
}

#[test]
fn custom_value_formats_allow_custom_field_names() {
    let source = r#"
        record Pair {
            left: Value,
            right: Value,
        }
        op Add<T: Integer>(left: T, right: T) -> (result: T) {
    meta: OpInfo { traits: [], memory: Known([]) },
            mnemonic: "add", storage: Pair { left: left, right: right },
             }
    "#;
    assert!(compile(source).is_ok());
}

#[test]
fn record_fields_do_not_double_as_text_configuration() {
    rejected(
        &changed_record(
            "record",
            "Iconst",
            "value: Int",
            "value: Int, text: IntegerConstant",
        ),
        "unknown data type",
    );
}
