//! Compile full operation contracts and reject malformed layouts and signatures.
use super::common::{self, compile};

const FORMATS: &str = include_str!("../../../../veloc/mir/defs/formats.ops");

fn definitions() -> String {
    [FORMATS, include_str!("../../../../veloc/mir/defs/mir.ops")]
        .join("\n")
        .lines()
        .filter(|line| !line.trim_start().starts_with("import "))
        .collect::<Vec<_>>()
        .join("\n")
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

#[test]
fn float_literals_require_a_float_result_domain() {
    common::rejected(
        &definitions().replacen(
            "op Fconst(value: Float) -> type(value)",
            "op Fconst(value: Float) -> ScalarInteger",
            1,
        ),
        "scalar float first result",
    );
}

const CALL_VALUE: &str = r#"
    struct ApplyValue {
        callee: Value,
        args: ValueList,
    }
    op Apply(move callee: Callable, move args: sequence(Value)) -> signature {
    meta: OpInfo { traits: OpTraits::MAY_TRAP, memory: MemoryEffect::UNKNOWN },
        mnemonic: "apply-value",
        storage: ApplyValue { callee: callee, args: args },
        signature: callable(callee),
        text: "{callee}({args})",
        }
"#;

#[test]
fn the_actual_mir_definitions_compile_deterministically() {
    let source = common::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../veloc/mir/defs/module.ops"),
    )
    .unwrap();
    let plan = source.plan().unwrap();
    assert!(plan.definitions().operation_count() > 0);
    let first = plan.generate();
    let second = plan.generate();
    assert_eq!(first.instructions, source.compile().unwrap().instructions);
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

const BINARY: &str = "type Reg = rust(\"crate::Reg\");
enum InstField { variants: [Imm(i64)] }
storage Operands { opcode: GenericOpcode, view: InstView, reader: InstRead, writer: InstBuild, register: Reg, attributes: InstField,  prefix: \"G_\"  }\nstruct Binary { dst: Reg, lhs: Reg, rhs: Reg }";
const ADD: &str = "op G_SUM<T: Integer>(lhs: T, rhs: T) -> (dst: T) { meta: OpInfo {}, storage: Binary { dst, lhs, rhs }, semantics: bv.add(lhs, rhs) }";

#[test]
fn packed_layout_contracts() {
    assert!(compile(&definitions()).is_ok());
    // Mutate the production schema: field types, operand groups, presence,
    // naming, and predication must all honor the same runtime layout contract.
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
        ("IntCompare", "args: values(2)", "args: values(3)"),
        ("VectorStoreStrided", "args: values(3)", "args: values(2)"),
        ("VectorScatter", "args: values(3)", "args: ValueList"),
        ("Shuffle", "args: values(2)", "args: values(3)"),
        ("Iconst", "value: Int,", ""),
        ("Iconst", "value: Int", "value: Int, unused: u32"),
        ("Load", "    flags: MemFlags,\n", ""),
        ("CallIndirect", "    sig_id: SigId,\n", ""),
        ("Unary", "arg: Value", "operand: Value"),
        ("Binary", "args: values(2)", "inputs: values(2)"),
        ("Ternary", "args: values(3)", "inputs: values(3)"),
        ("IntToPtr", "arg: Value", "operand: Value"),
        (
            "VectorOpWithExt",
            "ext: VectorExtData",
            "config: VectorExtData",
        ),
        (
            "VectorOpWithExt",
            "ext: VectorExtData",
            "ext: VectorMemOptions",
        ),
        (
            "VectorOpWithExt",
            "ext: VectorExtData",
            "ext: VectorExtData, hidden: u32",
        ),
    ] {
        common::rejected(
            &changed_record("struct", layout, from, to),
            "field contract",
        );
    }

    // Newly declared layouts are still free to choose their own field names.

    let source = r#"
        struct Pair {
            left: Value,
            right: Value,
        }
        op Add<T: Integer>(left: T, right: T) -> (result: T) {
    meta: OpInfo { traits: OpTraits::empty(), memory: MemoryEffect::NONE },
            mnemonic: "add", storage: Pair { left: left, right: right },
             }
    "#;
    assert!(compile(source).is_ok());
}

#[test]
fn text_projection_contracts() {
    // text projections cover every logical parameter once
    {
        for args in ["{lhs}", "{lhs}, {lhs}", "{lhs}, {missing}"] {
            common::rejected(
                &changed_record(
                    "op",
                    "IAdd",
                    "storage: Binary { args: [lhs, rhs] },",
                    &format!("storage: Binary {{ args: [lhs, rhs] }}, text: \"{args}\","),
                ),
                "",
            );
        }
        common::rejected(
            &changed_record("op", "Load", "{ptr}", "{ptr}, {offset}"),
            "offset",
        );
    }

    // typed text atoms do not accept incompatible fields
    {
        for atom in ["{ptr:integer}", "{ptr:float}", "{ptr:bytes}"] {
            common::rejected(&changed_record("op", "Load", "{ptr}", atom), "");
        }
        for named in [
            "[offset={offset}]",
            "offset={offset=-1}",
            "offset={offset=true}",
        ] {
            common::rejected(&changed_record("op", "Load", "offset={offset}", named), "");
        }
    }

    // compound property paths and optional values are checked
    {
        for (from, to) in [
            ("{mem.mask}", "{mem.unknown}"),
            ("{mem.evl}", "{mem.offset}"),
            ("{.mem.flags}", "{.mem.offset}"),
            (
                "offset={mem.offset}",
                "offset={mem.offset}[, mask={mem.mask}]",
            ),
        ] {
            common::rejected(&changed_record("op", "Gather", from, to), "");
        }
    }

    // record fields do not double as text configuration
    {
        common::rejected(
            &changed_record(
                "struct",
                "Iconst",
                "value: Int",
                "value: Int, text: IntegerConstant",
            ),
            "unknown data type",
        );
    }
}

#[test]
fn signature_sources() {
    // signature results require a typed signature source
    {
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
            common::rejected(&changed_record("op", op, from, to), "signature");
        }
    }

    // callable signature source requires a single named callable operand
    {
        compile(CALL_VALUE).unwrap();
        for (from, to, expected) in [
            (
                "signature: callable(callee),",
                "",
                "explicit signature source",
            ),
            (
                "callee: Callable",
                "callee: Type::PTR",
                "must be a Callable value operand",
            ),
            (
                "callable(callee)",
                "callable(args)",
                "must be a Callable value operand",
            ),
            (
                "callable(callee)",
                "callable(missing)",
                "must be a Callable value operand",
            ),
            (
                "callable(callee)",
                "callable()",
                "expected signature parameter",
            ),
            (
                "callable(callee)",
                "callable(callee, args)",
                "expected signature parameter",
            ),
            (
                "callable(callee)",
                "function(callee)",
                "must be a FuncId property",
            ),
            ("callable(callee)", "callee", "must be a SigId property"),
            (
                "-> signature",
                "-> ()",
                "signature source requires signature results",
            ),
        ] {
            let error = compile(&CALL_VALUE.replace(from, to)).err().unwrap();
            assert!(error.message.contains(expected), "{to}: {error}");
        }
    }
}

#[test]
fn output_plan_diagnostics() {
    // output errors are rejected before emission
    {
        let base = r#"
    struct Custom { arg: Value }
    op Example<T: Integer>(arg: T) -> T {
        meta: OpInfo { memory: MemoryEffect::NONE },
        mnemonic: "example", storage: Custom { arg: arg },
    }
    "#;
        for (source, message) in [
            (base.replace("mnemonic: \"example\"", "mnemonic: \"emit\""), "InstBuilder method"),
            (base.replace("storage: Custom { arg: arg },", "storage: Custom { arg: arg }, text: \"{missing}\","), "missing"),
            (format!("{base}\nstruct Alternate {{ arg: Value, extra: u32 }}\nlayout Alternate {{ format: fixed(Custom), text: \"{{arg}}, extra={{extra}}\", verify {{unknown > 0;
    }} }}"), "unknown expression name or operation"),
        ] {
            let source = common::source(&source);
            veloc_opgen::parse(&source).unwrap();
            let error = veloc_opgen::plan(&source).err().expect("invalid output plan");
            assert!(error.message.contains(message), "{error}");
        }
    }

    // property constraints are checked without an output plan
    {
        let source = common::source(
            "property Int { verify {unknown > 0;
    } }",
        );
        let error = veloc_opgen::parse(&source)
            .err()
            .expect("invalid property contract");
        assert!(
            error
                .message
                .contains("unknown expression name or operation"),
            "{error}"
        );
    }

    // invalid definitions fail before emission
    {
        for (source, message) in [
            (
                format!(
                    "{BINARY} {}",
                    ADD.replace("storage: Binary", "storage: Missing")
                ),
                "unknown operand format",
            ),
            (
                format!("{BINARY} {}", ADD.replace("rhs: T", "other: T")),
                "unknown input",
            ),
            (
                format!(
                    "{BINARY} {}",
                    ADD.replace("-> (dst: T)", "-> (dst: T, extra: T)")
                ),
                "every result requires a storage mapping",
            ),
            (
                format!("{BINARY} {}", ADD.replace("Integer", "Missing")),
                "Missing",
            ),
            (
                format!("{BINARY} {}", ADD.replace("rhs)", "missing)")),
                "unknown semantic value",
            ),
            (
                format!("{BINARY} {}", ADD.replace("Integer", "Float")),
                "floating-point",
            ),
            (
                format!(
                    "{BINARY} {}",
                    ADD.replace("semantics:", "flow: Call, semantics:")
                ),
                "flow requires a declared control enum",
            ),
            (format!("{BINARY} {ADD} {ADD}"), "duplicate op"),
            (format!("{} {ADD}", BINARY.replace("writer: InstBuild", "writer: InstRead")), "generated type names must be distinct"),
            (format!("{} {ADD}", BINARY.replace("opcode: GenericOpcode", "opcode: Type")), "conflicts with a declaration"),
            (
                "type Reg = rust(\"crate::Reg\");
enum InstField { variants: [Imm(i64)] }
storage Operands { opcode: GenericOpcode, view: InstView, reader: InstRead, writer: InstBuild, register: Reg, attributes: InstField,  } struct Bad { dst: Reg, dst: Reg }"
                    .into(),
                "duplicate field",
            ),
            (
                "type Reg = rust(\"crate::Reg\");
enum InstField { variants: [Imm(i64)] }
storage Operands { opcode: GenericOpcode, view: InstView, reader: InstRead, writer: InstBuild, register: Reg, attributes: InstField,  } struct Bad { values: sequence(Reg), dst: Reg } op Bad(dst: Type::I32, values: sequence(Value)) -> () { meta: OpInfo { memory: MemoryEffect::NONE }, storage: Bad { values, dst } }"
                    .into(),
                "only one trailing sequence",
            ),
            (
                "type Reg = rust(\"crate::Reg\");
enum InstField { variants: [Imm(i64)] }
storage Operands { opcode: GenericOpcode, view: InstView, reader: InstRead, writer: InstBuild, register: Reg, attributes: InstField,  } struct Bad { dst: Reg } layout Bad { lengths: [0] }"
                    .into(),
                "operand layouts",
            ),
            (
                "type Reg = rust(\"crate::Reg\");
enum InstField { variants: [Imm(i64)] }
storage Operands { opcode: GenericOpcode, view: InstView, reader: InstRead, writer: InstBuild, register: Reg, attributes: InstField,  } struct Bad { dst: Reg } layout Bad { lengths: [1, 1] }"
                    .into(),
                "operand layouts",
            ),
            (
                format!(
                    "{BINARY} {}",
                    ADD.replace("semantics:", "arity: 3, semantics:")
                ),
                "arity",
            ),
        ] {
            let error = veloc_opgen::parse(&common::source(&source))
                .err()
                .expect(&source);
            assert!(error.message.contains(message), "{source}\n{error}");
        }
    }

    // Text contracts are supported by the shared text compiler.
    {
        let source = common::source(&format!(
            "{BINARY} {}",
            ADD.replace("semantics:", "text: \"{lhs}, {rhs}\", semantics:")
        ));
        let generated = veloc_opgen::compile(&source).unwrap();
        assert!(!generated.text_parser.is_empty());
        assert!(!generated.text_printer.is_empty());
    }
}
