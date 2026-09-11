use veloc_isle::compile;

#[test]
fn memory_contracts_are_inherited_and_generated() {
    let generated = compile(
        r#"
        (def-template Load () (memory Read 8) (operands (def $dst) (use $ptr)))
        (def-inst Load64 (template Load))
        (def-inst Load32 (template Load) (memory Read 4))
        (def-pseudo-inst Store (memory Write 16) (operands (use $src)))
    "#,
        "x86_64",
    )
    .unwrap();
    for (name, shape) in [
        ("LOAD64", "Read, 8"),
        ("LOAD32", "Read, 4"),
        ("STORE", "Write, 16"),
    ] {
        let metadata = generated
            .split(&format!("TARGET_INST_{name}_METADATA:"))
            .nth(1)
            .unwrap()
            .split("};")
            .next()
            .unwrap();
        assert!(metadata.contains(&format!("memory: Some((veloc_lir::MemoryKind::{shape}))")));
    }
}

#[test]
fn invalid_memory_and_movable_accesses_are_rejected() {
    for body in [
        "(memory Read 0)",
        "(memory Read -1)",
        "(memory Unknown 4)",
        "(memory Read 4) (memory Read 8)",
        "(memory Read 4) (schedule 1)",
        "(memory Write 8) (schedule 1)",
    ] {
        assert!(
            compile(&format!("(def-inst Bad {body})"), "x86_64").is_err(),
            "{body}"
        );
    }
}
