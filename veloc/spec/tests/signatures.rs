mod common;
use common::compile;

const CALL_VALUE: &str = r#"
    struct ApplyValue {
        callee: Value,
        args: ValueList,
    }
    op Apply(move callee: Callable, move args: values) -> signature {
    meta: OpInfo { traits: [MAY_TRAP], memory: Unknown },
        mnemonic: "apply-value",
        storage: ApplyValue { callee: callee, args: args },
        signature: callable(callee),
        text: "{callee}({args})",
        }
"#;

#[test]
fn callable_signature_source_requires_a_single_named_callable_operand() {
    compile(CALL_VALUE).unwrap();
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
        let error = compile(&CALL_VALUE.replace(from, to)).err().unwrap();
        assert!(error.message.contains(expected), "{to}: {error}");
    }
}
