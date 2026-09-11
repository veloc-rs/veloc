use veloc_isle::compile;

#[test]
fn scheduling_metadata_is_inherited_and_can_be_overridden() {
    let generated = compile(
        r#"
      (def-template Add () (schedule 1) (clobbers EFLAGS)
        (operands (def (tied $dst $lhs)) (use $rhs)) (encode (byte 0x90)))
      (def-inst Fast (template Add))
      (def-inst Slow (schedule 3) (template Add))
      (def-inst Barrier (encode (byte 0x90)))
    "#,
        "x86_64",
    )
    .unwrap();
    assert!(generated.contains("latency: 1, writes_flags: true"));
    assert!(generated.contains("latency: 3, writes_flags: true"));
    assert!(generated.contains("schedule: None"));
}

#[test]
fn invalid_or_incomplete_scheduling_contracts_are_rejected() {
    for body in [
        "(schedule 0)",
        "(schedule -1)",
        "(schedule 1) (schedule 2)",
        "(schedule 1) (operands (block $b))",
        "(schedule 1) (clobbers UNKNOWN)",
    ] {
        let source = format!("(def-inst Bad {body} (encode (byte 0x90)))");
        assert!(compile(&source, "x86_64").is_err(), "{source}");
    }
}
