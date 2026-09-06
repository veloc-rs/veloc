use veloc_opgen::{Generated, compile_mir};

const PAIR: &str = r#"
format Pair {
    fields: [op(Opcode), inputs(values(2))],
    opcode: dynamic(op)
}
op Difference<T: Integer>(left: T, right: T) -> (result: T) {
    mnemonic: "difference",
    storage: Pair { inputs: [left, right] },
    memory: NONE
}
"#;

fn assert_order(text: &str, first: &str, second: &str) {
    let first = text
        .find(first)
        .expect("first projection was not generated");
    let second = text
        .find(second)
        .expect("second projection was not generated");
    assert!(first < second, "projection order was reversed:\n{text}");
}

fn rejected(source: &str, expected: &str) {
    let result = std::panic::catch_unwind(|| compile_mir(source))
        .expect("invalid text definitions must produce a diagnostic, not panic");
    let error = result.err().expect("invalid text definition was accepted");
    assert!(error.message.contains(expected), "{error}");
    assert!(error.line > 0 && error.column > 0);
}

fn alternate(fields: &str, projection: &str) -> String {
    format!(
        "{PAIR}\nlayout Extended {{
            fields: [{fields}], opcode: dynamic(op),
            format: arity(inputs, [Pair]), text: {projection}
        }}"
    )
}

#[test]
fn implicit_text_uses_logical_order_in_both_directions() {
    let Generated {
        text_parser,
        text_printer,
        ..
    } = compile_mir(PAIR).unwrap();
    assert!(text_parser.contains(
        "let _p0 = <crate::Value as super::atom::AtomCodec>::parse(self, _core[0], ty)?;"
    ));
    assert!(text_parser.contains(
        "let _p1 = <crate::Value as super::atom::AtomCodec>::parse(self, _core[1], ty)?;"
    ));
    assert_order(
        &text_printer,
        "AtomCodec>::print(self, f, &_p0, ty)",
        "AtomCodec>::print(self, f, &_p1, ty)",
    );
    assert!(text_parser.contains("inputs: [_p0, _p1]"));
}

#[test]
fn generated_atoms_and_pools_use_typed_contracts_not_concrete_helper_names() {
    let source = [
        include_str!("../../mir/defs/formats.ops"),
        include_str!("../../mir/defs/mir.ops"),
    ]
    .join("\n");
    let output = compile_mir(&source).unwrap();
    for codec in [
        "crate::Value",
        "crate::FuncId",
        "crate::SigId",
        "crate::BlockCall",
        "crate::IntCC",
        "crate::FloatCC",
        "crate::Intrinsic",
        "crate::StackSlot",
        "bool",
        "super::atom::IntegerBits",
        "super::atom::FloatBits",
        "super::atom::Bytes",
        "super::atom::Values",
        "super::atom::Successors",
        "super::atom::Decimal<u8>",
        "super::atom::Decimal<u32>",
        "super::atom::Decimal<i32>",
    ] {
        assert!(
            output
                .text_parser
                .contains(&format!("<{codec} as super::atom::AtomCodec>::parse(")),
            "missing reader for {codec}"
        );
        assert!(
            output
                .text_printer
                .contains(&format!("<{codec} as super::atom::AtomCodec>::print(")),
            "missing writer for {codec}"
        );
    }
    for key in [
        "PtrIndexImmId",
        "VectorExtId",
        "VectorMemExtId",
        "ConstantPoolId",
    ] {
        assert!(output.text_parser.contains(&format!(
            "<crate::inst::{key} as crate::dfg::PoolKey>::insert("
        )));
        assert!(output.text_printer.contains(&format!(
            "<crate::inst::{key} as crate::dfg::PoolKey>::get("
        )));
    }
    for helper in [
        "self.value(",
        "self.func_ref(",
        "self.float_bits(",
        "self.integer_bits(",
        "self.vf(",
        "fmt_float_bits",
        "fmt_integer_bits",
        "intern_ptr_imm",
        "intern_vector_ext",
        "make_vector_mem_ext",
        "make_constant_pool_data",
        "ConstantPoolData::",
    ] {
        assert!(
            !output.text_parser.contains(helper),
            "concrete reader coupling: {helper}"
        );
        assert!(
            !output.text_printer.contains(helper),
            "concrete writer coupling: {helper}"
        );
    }
}

#[test]
fn explicit_text_order_changes_both_directions_without_changing_storage_or_builder() {
    let source = PAIR.replace(
        "memory: NONE",
        "text: Text { args: [right, left] }, memory: NONE",
    );
    let output = compile_mir(&source).unwrap();
    assert!(output.text_parser.contains(
        "let _p1 = <crate::Value as super::atom::AtomCodec>::parse(self, _core[0], ty)?;"
    ));
    assert!(output.text_parser.contains(
        "let _p0 = <crate::Value as super::atom::AtomCodec>::parse(self, _core[1], ty)?;"
    ));
    assert_order(
        &output.text_printer,
        "AtomCodec>::print(self, f, &_p1, ty)",
        "AtomCodec>::print(self, f, &_p0, ty)",
    );
    assert!(output.text_parser.contains("inputs: [_p0, _p1]"));
    assert!(output.opcodes.contains(
        "pub fn difference(&mut self, left: crate::Value, right: crate::Value) -> crate::Value"
    ));
    assert!(output.opcodes.contains("inputs: [left, right]"));
}

#[test]
fn named_property_defaults_drive_parsing_and_canonical_printing_together() {
    let source = r#"
        format Immediate {
            fields: [op(Opcode), value(Value), displacement(i32)], opcode: dynamic(op)
        }
        op Offset(arg: PTR, @amount: i32) -> (result: PTR) {
            mnemonic: "offset", storage: Immediate { value: arg, displacement: amount },
            text: Text { args: [arg], named: [default(amount, 7)] }, memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(
        output
            .text_parser
            .contains("reject_unknown_named(&_named, &[\"amount\"])?;")
    );
    assert!(output.text_parser.contains("None => 7"));
    assert!(output.text_parser.contains("displacement: _p1"));
    assert!(output.text_printer.contains("if _p1 != 7"));
    assert!(output.text_printer.contains("f.write_str(\"amount=\")?;"));
    assert!(!output.text_printer.contains("displacement="));
}

#[test]
fn checked_alternate_fields_are_generated_in_both_directions() {
    let source = alternate(
        "op(Opcode), inputs(ValueList), tag(u32)",
        "Text { args: [inputs], named: [tag] }",
    );
    let output = compile_mir(&source).unwrap();
    assert!(
        output
            .text_parser
            .contains("split_core_and_named(text, Some(2))?")
    );
    assert!(output.text_parser.contains("InstructionData::Extended"));
    assert!(
        output
            .text_parser
            .contains("named_value(&_named, \"tag\")?")
    );
    // A fallible text lookup must not be duplicated inside a map_err closure,
    // where `?` would require that closure to return Result instead of ParseError.
    assert_eq!(
        output
            .text_parser
            .matches("named_value(&_named, \"tag\")?")
            .count(),
        1
    );
    assert!(output.text_printer.contains("InstructionData::Extended"));
    assert!(output.text_printer.contains("f.write_str(\"tag=\")?;"));
    assert!(output.text_printer.contains(".len() != 2"));
}

#[test]
fn incomplete_or_ambiguous_alternates_return_definition_errors() {
    let fields = "op(Opcode), inputs(ValueList), tag(u32)";
    rejected(
        &alternate(fields, "Text { args: [inputs] }"),
        "does not consume `tag`",
    );
    rejected(
        &alternate(fields, "Text { args: [inputs], named: [default(tag, 0)] }"),
        "required named fields",
    );
    rejected(
        &alternate(fields, "Text { args: [inputs], named: [optional(tag)] }"),
        "optional(Value)",
    );
    let source = alternate(fields, "Text { args: [inputs], named: [tag] }");
    let second = source.split("layout Extended").nth(1).unwrap();
    rejected(
        &format!("{source}\nlayout Another{second}"),
        "multiple text alternatives",
    );
}

#[test]
fn unsupported_alternate_storage_groups_fail_before_rust_emission() {
    let source = format!(
        "{PAIR}\nlayout Extended {{
            fields: [op(Opcode), inputs(values(2)), tag(u32)],
            opcode: dynamic(op), format: fixed(Pair),
            text: Text {{ args: [inputs], named: [tag] }}
        }}"
    );
    rejected(&source, "named storage types");
}
