use veloc_isle::target::compile;

#[test]
fn control_metadata_is_generated_for_instructions_templates_and_pseudos() {
    let generated = compile(
        r#"
        (def-template Conditional () (flow Branch) (operands (block $target)))
        (def-inst Branch (template Conditional))
        (def-inst Jump (template Conditional) (flow Jump))
        (def-inst Ret (flow Return))
        (def-inst Call (flow Call))
        (def-inst Trap (flow Trap))
        (def-pseudo-inst TailJump (flow Jump) (operands (use $callee)))
        (def-inst Normal (encode (byte 0x90)))
        "#,
        "x86_64",
    )
    .unwrap();
    for flow in ["Next", "Branch", "Jump", "Return", "Call", "Trap"] {
        assert!(generated.contains(&format!("flow: veloc_lir::ControlFlow::{flow}")));
    }
    for (name, flow) in [("BRANCH", "Branch"), ("JUMP", "Jump"), ("TAILJUMP", "Jump")] {
        let metadata = generated
            .split(&format!("TARGET_INST_{name}_METADATA:"))
            .nth(1)
            .unwrap();
        let metadata = metadata.split("};").next().unwrap();
        assert!(
            metadata.contains(&format!("ControlFlow::{flow}")),
            "{metadata}"
        );
    }
}

#[test]
fn invalid_control_and_movable_control_are_rejected() {
    for body in [
        "(flow Unknown)",
        "(flow Next) (flow Return)",
        "(flow Return) (schedule 1)",
        "(flow Call) (schedule 1)",
        "(flow Trap) (schedule 1)",
        "(flow Branch) (schedule 1)",
        "(flow Jump) (schedule 1)",
    ] {
        let source = format!("(def-inst Bad {body})");
        assert!(compile(&source, "x86_64").is_err(), "{source}");
    }
}
