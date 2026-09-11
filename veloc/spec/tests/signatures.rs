mod common;
use common::compile;

const CALL_VALUE: &str = r#"
    format ApplyValue {
        fields: [opcode(Opcode), callee(Value), args(ValueList)],
        opcode: dynamic(opcode)
    }
    op Apply(callee: Callable, args: values) -> signature {
        mnemonic: "apply-value",
        storage: ApplyValue { callee: callee, args: args },
        signature: callable(callee),
        text: "{callee}({args})",
        moves: [callee, args],
        control: call(callee, args),
        memory: UNKNOWN
    }
"#;

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
        let error = compile(&source.replace(from, to)).err().unwrap();
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
        let error = compile(&CALL_VALUE.replace(from, to)).err().unwrap();
        assert!(error.message.contains("control interface"), "{to}: {error}");
    }
    for legacy in [
        "resume(callee, args)",
        "invoke(callee, args)",
        "cancel(callee)",
    ] {
        let error = compile(&CALL_VALUE.replace("call(callee, args)", legacy))
            .err()
            .unwrap();
        assert!(
            error.message.contains("invalid control interface"),
            "{error}"
        );
    }
}
