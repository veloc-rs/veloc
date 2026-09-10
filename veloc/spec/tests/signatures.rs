mod common;
use common::compile_mir;

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

fn artifacts(output: veloc_opgen::Generated) -> [String; 7] {
    [
        output.types,
        output.type_rules,
        output.builders,
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
            .type_rules
            .contains("results[1] must have more bits per lane than operands[0]")
    );
    let both_named = source.replace("(T, wider: I64)", "(result: T, wider: I64)");
    assert_eq!(
        artifacts(output),
        artifacts(compile_mir(&both_named).unwrap())
    );
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
    assert!(output.type_rules.contains("C::Integer.accepts(results[0])"));
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

const CALL_VALUE: &str = r#"
    format ApplyValue {
        fields: [opcode(Opcode), callee(Value), args(ValueList)],
        opcode: dynamic(opcode)
    }
    op Apply(callee: Callable, args: values) -> signature {
        mnemonic: "apply-value",
        storage: ApplyValue { callee: callee, args: args },
        signature: callable(callee),
        text: Text { args: [apply(callee, args)] },
        moves: [callee, args],
        control: call(callee, args),
        memory: UNKNOWN
    }
"#;

#[test]
fn callable_signature_infers_results_and_checks_arguments_and_results() {
    let output = compile_mir(CALL_VALUE).unwrap();
    assert!(
        output
            .instructions
            .contains("dfg.values().get(*source).and_then(|value| value.ty.as_callable())")
    );
    assert!(!output.instructions.contains("SignatureRef"));
    assert!(!output.instructions.contains("CallInfo"));
    assert!(
        output
            .opcodes
            .contains("pub const fn has_signature(self) -> bool")
    );
    assert!(
        output
            .opcodes
            .contains("pub const fn has_control(self) -> bool")
    );
    assert!(output.opcodes.contains("Self::Apply"));
    assert!(!output.instructions.contains("Control::"));
    assert!(output.validation.contains(".as_callable().ok_or_else"));
    for expected in [
        "signature.params.iter().copied()",
        "self.dfg.inst_results(_inst), signature.returns.iter().copied()",
    ] {
        assert!(output.validation.contains(expected), "{expected}");
    }
    assert!(output.type_rules.contains("operands[0].is_callable()"));
}

#[test]
fn control_classification_uses_definitions_without_operand_wrappers() {
    for (source, expected) in [
        (PAIR.to_owned(), "false"),
        (CALL_VALUE.to_owned(), "true"),
        (
            format!("{PAIR}\n{CALL_VALUE}"),
            "matches!(self, Self::Apply)",
        ),
    ] {
        let output = compile_mir(&source).unwrap();
        assert!(output.opcodes.contains(&format!(
            "pub const fn has_control(self) -> bool {{ {expected} }}"
        )));
        assert!(output.opcodes.contains(&format!(
            "pub const fn has_signature(self) -> bool {{ {expected} }}"
        )));
        assert!(!output.instructions.contains("pub fn control("));
    }
}

#[test]
fn callable_signature_source_requires_a_single_named_callable_operand() {
    // Omit control so these failures exercise the signature contract directly.
    let source = CALL_VALUE.replace("control: call(callee, args),", "");
    for (from, to, expected) in [
        (
            "signature: callable(callee),",
            "",
            "explicit signature source",
        ),
        (
            "callee: Callable",
            "callee: PTR",
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
        let error = compile_mir(&source.replace(from, to)).err().unwrap();
        assert!(error.message.contains(expected), "{to}: {error}");
    }
}

#[test]
fn callable_control_preserves_result_terminator_and_effect_contracts() {
    for (from, to) in [
        ("-> signature", "-> ()"),
        ("callee: Callable", "callee: PTR"),
        ("memory: UNKNOWN", "memory: NONE"),
        ("memory: UNKNOWN", "traits: [TERMINATOR], memory: UNKNOWN"),
        ("call(callee, args)", "tail_call_value(callee, args)"),
        ("call(callee, args)", "call(args, callee)"),
    ] {
        let error = compile_mir(&CALL_VALUE.replace(from, to)).err().unwrap();
        assert!(error.message.contains("control interface"), "{to}: {error}");
    }
    for legacy in [
        "resume(callee, args)",
        "invoke(callee, args)",
        "cancel(callee)",
    ] {
        let error = compile_mir(&CALL_VALUE.replace("call(callee, args)", legacy))
            .err()
            .unwrap();
        assert!(
            error.message.contains("invalid control interface"),
            "{error}"
        );
    }
}
