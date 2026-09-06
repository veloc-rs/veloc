mod common;
use common::compile_mir;
use veloc_opgen::Error;

const PAIR: &str = r#"
format Pair {
    fields: [op(Opcode), args(values(2))],
    opcode: dynamic(op)
}
op PairAdd<T: Integer>(left: T, right: T) -> (result: T) {
    mnemonic: "pair-add",
    storage: Pair { args: [left, right] },
    memory: NONE
}
"#;

const LOAD: &str = r#"
format Load {
    fields: [ptr(Value), offset(u32), flags(MemFlags)],
    opcode: fixed(Load)
}
op Load(ptr: PTR, @offset: u32, @flags: MemFlags) -> (result: Any) {
    mnemonic: "load",
    storage: Load { ptr: ptr, offset: offset, flags: flags },
    text: Text { args: [ptr], named: [default(offset, 0)], flags: flags },
    traits: [MAY_TRAP], memory: HEAP_READ
}
"#;

const INDIRECT_CALL: &str = r#"
format CallIndirect {
    fields: [ptr(Value), args(ValueList), sig_id(SigId)],
    opcode: fixed(CallIndirect)
}
op CallIndirect(@sig_id: SigId, ptr: PTR, args: values) -> signature {
    mnemonic: "call-indirect",
    storage: CallIndirect { ptr: ptr, args: args, sig_id: sig_id },
    text: Text { args: [invoke(ptr, args, sig_id)] },
    signature: sig_id,
    traits: [MAY_TRAP], memory: UNKNOWN
}
"#;

fn rejected(source: &str) -> Error {
    let error = match compile_mir(source) {
        Ok(_) => panic!("invalid signature was accepted:\n{source}"),
        Err(error) => error,
    };
    assert!(error.line > 0 && error.column > 0, "{error}");
    error
}

fn artifacts(output: veloc_opgen::Generated) -> [String; 9] {
    [
        output.encoding,
        output.builtins,
        output.scalars,
        output.formats,
        output.types,
        output.opcodes,
        output.instructions,
        output.text_parser,
        output.text_printer,
    ]
}

#[test]
fn result_names_do_not_change_generated_artifacts() {
    for (named, alternatives) in [
        ("(result: T)", vec!["T", "(T)", "(answer: T)"]),
        (
            "(result: T, overflow: BOOL)",
            vec!["(T, BOOL)", "(T, overflow: BOOL)", "(result: T, BOOL)"],
        ),
        (
            "(result: shape(T, Integer))",
            vec!["shape(T, Integer)", "(shape(T, Integer))"],
        ),
    ] {
        let reference = artifacts(compile_mir(&PAIR.replace("(result: T)", named)).unwrap());
        for results in alternatives {
            let source = PAIR.replace("(result: T)", results);
            assert_eq!(
                artifacts(compile_mir(&source).unwrap()),
                reference,
                "{results}"
            );
        }
    }
    let reference = artifacts(compile_mir(LOAD).unwrap());
    assert_eq!(
        artifacts(compile_mir(&LOAD.replace("(result: Any)", "Any")).unwrap()),
        reference
    );
}

#[test]
fn named_results_keep_their_position_among_anonymous_results() {
    let source = PAIR
        .replace("(result: T)", "(T, wider: I64)")
        .replace("memory: NONE", "where: [wider(left, wider)], memory: NONE");
    let output = compile_mir(&source).unwrap();
    assert!(
        output
            .types
            .contains("results[1] must have more bits per lane than operands[0]")
    );
    let both_named = source.replace("(T, wider: I64)", "(result: T, wider: I64)");
    assert_eq!(
        artifacts(output),
        artifacts(compile_mir(&both_named).unwrap())
    );
}

#[test]
fn anonymous_results_have_no_synthetic_names() {
    for reference in ["result", "result0"] {
        let source = PAIR.replace("(result: T)", "Integer").replace(
            "memory: NONE",
            &format!("where: [wider(left, {reference})], memory: NONE"),
        );
        assert!(
            rejected(&source)
                .message
                .contains("missing operand or result")
        );
    }
    // An operand may use `result`; anonymous returns do not reserve that name.
    let source = PAIR.replace("(result: T)", "T").replace("left", "result");
    assert!(compile_mir(&source).is_ok());
}

#[test]
fn result_only_type_variables_and_nested_type_patterns_still_bind() {
    let source = r#"
        format Empty { fields: [opcode(Opcode)], opcode: dynamic(opcode) }
        op Pair<T: Integer>() -> T {
            mnemonic: "pair", storage: Empty {}, memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.types.contains("C::Integer.accepts(results[0])"));
    // Multiple explicit results are valid signatures, but not supported by the
    // current field-builder projection. Check their binding at the model layer.
    common::parse(&source.replace("-> T", "-> (T, T)")).unwrap();
    let vector = PAIR
        .replace("T: Integer", "T: Vector")
        .replace("(result: T)", "element(T)");
    assert_eq!(
        artifacts(compile_mir(&vector).unwrap()),
        artifacts(
            compile_mir(&vector.replace("-> element(T)", "-> (result: element(T))")).unwrap()
        )
    );
}

#[test]
fn bare_result_types_leave_the_operation_body_unconsumed() {
    for result in ["T", "BOOL", "shape(T, Integer)"] {
        let source = PAIR.replace("(result: T)", &format!("{result} // result type\n"));
        assert!(compile_mir(&source).is_ok(), "{result}");
    }
    for result in [
        "",
        "(T",
        "T, BOOL",
        "(T BOOL)",
        "(result:)",
        "(shape(T, Integer): T)",
        "(@result: T)",
        "([T])",
        "result: T",
    ] {
        let source = PAIR.replace("(result: T)", result);
        rejected(&source);
    }
}

#[test]
fn a_generic_is_bound_once_and_shared_by_named_operands_and_results() {
    let output = compile_mir(PAIR).unwrap();
    assert!(output.types.contains("operands[1] == operands[0]"));
    assert!(output.types.contains("results[0] == operands[0]"));
    assert!(output.opcodes.contains(
        "pub fn pair_add(&mut self, left: crate::Value, right: crate::Value) -> crate::Value"
    ));
    assert!(output.opcodes.contains("args: [left, right]"));
}

#[test]
fn variadic_calls_preserve_the_statically_checked_pointer_prefix() {
    let output = compile_mir(INDIRECT_CALL).unwrap();
    assert!(output.types.contains("operands[0] == Type::PTR"));
    assert!(output.types.contains("Ok(super::ResultTypes::Signature)"));
    assert!(
        output
            .opcodes
            .contains("Ok((crate::opspec::ResultTypes::Signature, true))")
    );
    // The signature property is not an SSA operand in the type contract.
    assert!(!output.types.contains("SigId"));
}

#[test]
fn successors_preserve_the_statically_checked_branch_condition() {
    let source = r#"
        format Br {
            fields: [condition(Value), then_dest(BlockCall), else_dest(BlockCall)],
            opcode: fixed(Br)
        }
        op Br(condition: BOOL, then_dest: successor, else_dest: successor) -> () {
            mnemonic: "br",
            storage: Br { condition: condition, then_dest: then_dest, else_dest: else_dest },
            traits: [TERMINATOR], memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.types.contains("operands[0] == Type::BOOL"));
    assert!(output.types.contains("if !results.is_empty()"));
    assert!(
        output
            .opcodes
            .contains("ResultTypes::Inferred(Default::default()), true")
    );
}

#[test]
fn a_fixed_operand_after_a_variadic_group_is_an_explicit_definition_error() {
    let source = INDIRECT_CALL.replace(
        "@sig_id: SigId, ptr: PTR, args: values",
        "@sig_id: SigId, args: values, ptr: PTR",
    );
    let error = rejected(&source);
    assert!(error.message.contains("variadic"), "{error}");
}

#[test]
fn operand_and_result_names_must_be_unique() {
    for source in [
        PAIR.replace("right: T", "left: T"),
        PAIR.replace("result: T", "left: T"),
        PAIR.replace("result: T", "result: T, result: BOOL"),
        LOAD.replace("@offset: u32", "@ptr: u32"),
    ] {
        let error = rejected(&source);
        assert!(error.message.contains("duplicate"), "{error}");
    }
}

#[test]
fn generic_declarations_and_type_domains_are_checked() {
    for source in [
        PAIR.replace("T: Integer", "T: Integer, T: Float"),
        PAIR.replace("T: Integer", "T: Missing"),
        PAIR.replace("right: T", "right: Undeclared"),
        PAIR.replace("T: Integer", "T: ScalarInteger")
            .replace("right: T", "right: element(T)"),
        PAIR.replace("T: Integer", "T: Vector")
            .replace("right: T", "right: vector(T)"),
        PAIR.replace("T: Integer", "T: ScalarInteger")
            .replace("right: T", "right: shape(T, Vector)"),
    ] {
        rejected(&source);
    }
}

#[test]
fn storage_packing_covers_every_field_and_logical_parameter_once() {
    for source in [
        PAIR.replace("args: [left, right]", ""),
        PAIR.replace("args: [left, right]", "args: [left, left]"),
        PAIR.replace("args: [left, right]", "args: [left, unknown]"),
        PAIR.replace("args: [left, right]", "args: [left, right], extra: left"),
        PAIR.replace("right: T", "right: T, unused: T"),
        LOAD.replace(
            "ptr: ptr, offset: offset, flags: flags",
            "ptr: ptr, offset: offset",
        ),
    ] {
        rejected(&source);
    }
}

#[test]
fn properties_must_match_their_declared_storage_type_and_role() {
    let output = compile_mir(LOAD).unwrap();
    assert!(output.types.contains("operands[0] == Type::PTR"));
    assert!(output.opcodes.contains(
        "pub fn load(&mut self, ptr: crate::Value, offset: u32, flags: crate::MemFlags, ty: crate::Type) -> crate::Value"
    ));
    for source in [
        LOAD.replace("@offset: u32", "@offset: i32"),
        LOAD.replace("@flags: MemFlags", "@flags: bool"),
        LOAD.replace("@offset: u32", "offset: I32"),
        LOAD.replace("ptr: PTR", "@ptr: u32"),
        INDIRECT_CALL.replace("args: values", "args: successor"),
    ] {
        rejected(&source);
    }
}

#[test]
fn legacy_type_lists_and_position_based_relations_are_not_accepted() {
    for source in [
        "types OLD { operands: [], results: [] }".to_owned(),
        "op Old { mnemonic: \"old\", format: Pair, types: OLD, traits: [], memory: NONE }"
            .to_owned(),
        PAIR.replace("memory: NONE", "types: OLD, memory: NONE"),
        PAIR.replace("right: T", "right: same(T)"),
        PAIR.replace("left: T", "left: bind(T, Integer)"),
        PAIR.replace(
            "memory: NONE",
            "where: [wider(operand(0), result(0))], memory: NONE",
        ),
    ] {
        rejected(&source);
    }
}
