//! Target definitions from parsing through instruction selection and emission.
use veloc_isle::target::{Def, MatchKind, Pattern, compile, parse};

#[test]
fn compile_select_rule_v2_generates_target_inst_and_match_arm() {
    let input = r#"
        (def-inst X86Ret
          (operands)
          (encode (byte 0xC3)))

        (select-rule
          (match (G_RET @n))
          (emit (X86Ret))
          (covers (@n))
          (cost 1))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("pub enum TargetInst"));
    assert!(output.contains("X86Ret"));
    assert!(output.contains("GenericOpcode::G_RET"));
    assert!(output.contains("TargetInst::X86Ret.as_u32()"));
}

#[test]
fn compile_for_block_syntax_expands_table_rows() {
    let input = r#"
        for (reg_name) in
          R10
          R11
        do
        (def-reg {{reg_name}} (size 64) (hw-enc 10))
        end
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("REG_R10"));
    assert!(output.contains("REG_R11"));
}

#[test]
fn compile_for_block_syntax_expands_multi_column_rows() {
    let input = r#"
        for (reg_name, size, enc) in
          RAX, 64, 0
          RBX, 64, 3
        do
        (def-reg {{reg_name}} (size {{size}}) (hw-enc {{enc}}))
        end
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("REG_RAX"));
    assert!(output.contains("REG_RBX"));
}

#[test]
fn compile_for_block_syntax_supports_nested_blocks() {
    let input = r#"
        for (outer) in
          RAX
          RBX
        do
        for (inner) in
          8
          16
        do
        (def-reg {{outer}}_{{inner}} (size {{inner}}) (hw-enc 0))
        end
        end
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("REG_RAX_8"));
    assert!(output.contains("REG_RAX_16"));
    assert!(output.contains("REG_RBX_8"));
    assert!(output.contains("REG_RBX_16"));
}

#[test]
fn compile_for_block_placeholder_replacement_is_explicit() {
    let input = r#"
        for (reg) in
          RAX
        do
        (def-abi TestAbi
          (arch X86_64)
          (stack (align 16))
          (classifier target_reg))
        (def-reg {{reg}}_alias (size 64) (hw-enc 0))
        end
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("classifier: Some(\"target_reg\")"));
    assert!(output.contains("REG_RAX_alias"));
}

#[test]
fn compile_for_block_missing_do_reports_error() {
    let input = r#"
        for (reg, size) in
          RAX, 64
        end
    "#;

    let err = compile(input, "x86_64").expect_err("compile should reject missing do");
    assert!(err.contains("missing `do` for `for` block"));
}

#[test]
fn compile_for_block_missing_end_reports_error() {
    let input = r#"
        for (reg) in
          RAX
        do
        (def-reg {{reg}} (size 64) (hw-enc 0))
    "#;

    let err = compile(input, "x86_64").expect_err("compile should reject missing end");
    assert!(err.contains("missing `end` for `for` block"));
}

#[test]
fn compile_for_block_tuple_arity_mismatch_reports_error() {
    let input = r#"
        for (reg, size) in
          RAX
        do
        (def-reg {{reg}} (size {{size}}) (hw-enc 0))
        end
    "#;

    let err = compile(input, "x86_64").expect_err("compile should reject bad tuple arity");
    assert!(err.contains("invalid `for` tuple"));
}

#[test]
fn compile_for_block_old_syntax_is_rejected() {
    let input = r#"
        for reg
          RAX
        do
        (def-reg {{reg}} (size 64) (hw-enc 0))
        end
    "#;

    let err = compile(input, "x86_64").expect_err("compile should reject old syntax");
    assert!(err.contains("expected `for (vars) in`"));
}

#[test]
fn compile_rel32_emit_generates_block_fixup() {
    let input = r#"
        (def-inst X86Jmp
          (operands (block $target))
          (encode
            (byte 0xE9)
            (rel32 $target)))

        (select-rule
          (match (G_BR $target @n))
          (emit (X86Jmp $target))
          (covers (@n))
          (cost 1))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("emitter.add_block_rel32_fixup"));
    assert!(output.contains("InstField::Block(target)"));
}

#[test]
fn compile_def_abi_generates_descriptor_constant() {
    let input = r#"
        (def-reg RAX (size 64) (hw-enc 0))
        (def-reg RDX (size 64) (hw-enc 2))
        (def-reg RBP (size 64) (hw-enc 5))
        (def-reg RSI (size 64) (hw-enc 6))
        (def-reg RDI (size 64) (hw-enc 7))
        (def-reg R8  (size 64) (hw-enc 8))
        (def-reg R9  (size 64) (hw-enc 9))

        (def-abi X86_64SystemV
          (arch X86_64)
          (stack
            (align 16)
            (incoming-base RBP 16)
            (outgoing-slot 8 8))
          (args
            (class Integer (regs RDI RSI RDX RCX R8 R9)))
          (returns
            (class Integer (regs RAX RDX)))
          (classifier x86_64_sysv_classifier))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("pub static ABI_X86_64SYSTEMV: AbiDescriptor"));
    assert!(output.contains("AbiStackDescriptor"));
    assert!(output.contains("classifier: Some(\"x86_64_sysv_classifier\")"));
}

#[test]
fn compile_def_abi_rejects_unknown_arch() {
    let input = r#"
        (def-abi WeirdAbi
          (arch Sparc64)
          (stack (align 16)))
    "#;

    let err = compile(input, "x86_64").expect_err("compile should reject unknown ABI arch");
    assert!(err.contains("unsupported ABI architecture `Sparc64`"));
}

#[test]
fn compile_def_reg_generates_reserved_and_special_role_constants() {
    let input = r#"
        (def-reg RAX (size 64) (hw-enc 0))
        (def-reg RSP (size 64) (hw-enc 4) (reserved) (role stack-pointer))
        (def-reg RBP (size 64) (hw-enc 5) (reserved) (role frame-pointer))
        (def-regclass GPR64 (RAX RSP RBP))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("pub const RESERVED_REGS: &[Reg] = &[REG_RSP, REG_RBP];"));
    assert!(output.contains("pub const SPECIAL_REG_STACK_POINTER: Reg = REG_RSP;"));
    assert!(output.contains("pub const SPECIAL_REG_FRAME_POINTER: Reg = REG_RBP;"));
    assert!(output.contains("pub const REGCLASS_GPR64_ALLOCATABLE: &[Reg] = &[REG_RAX];"));
}

#[test]
fn compile_def_inst_generates_operand_constraint_metadata() {
    let input = r#"
        (def-reg RCX (size 64) (hw-enc 1))

        (def-inst X86Shl32Cl
          (operands (def (tied $dst $src1)) (use (fixed RCX $count)))
          (clobbers EFLAGS)
          (encode (byte 0x90)))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("pub const TARGET_INST_X86SHL32CL_METADATA: TargetInstMetadata"));
    assert!(output.contains("TiedOperandConstraint { result: 0, use_operand: 1 }"));
    assert!(output.contains("FixedUseConstraint { use_operand: 0, reg: REG_RCX }"));
    assert!(output.contains("clobbers: &[\"EFLAGS\"]"));
    assert!(output.contains("pub fn target_inst_metadata(opcode: TargetInst)"));
}

#[test]
fn compile_select_rules_generate_generic_operand_constraint_metadata() {
    let input = r#"
        (def-reg RCX (size 64) (hw-enc 1))

        (def-inst X86Add32
          (operands (def (tied $dst $src1)) (use $src2))
          (encode (byte 0x90)))

        (def-inst X86Shl32Cl
          (operands (def (tied $dst $src1)) (use (fixed RCX $count)))
          (encode (byte 0x90)))

        (select-rule
          (match (schema BinaryReg G_ADD (dst (GPR32 $dst)) (lhs (GPR32 $x)) (rhs (GPR32 $y)) @n))
          (emit (X86Add32 $y $x))
          (covers (@n))
          (cost 1))

        (select-rule
          (match (schema BinaryReg G_SHL (dst (GPR32 $dst)) (lhs (GPR32 $x)) (rhs (GPR32 $y)) @n))
          (emit (X86Shl32Cl $y $x))
          (covers (@n))
          (cost 1))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(!output.contains("pub const GENERIC_INST_G_ADD_METADATA: GenericInstMetadata"));
    assert!(output.contains("pub const GENERIC_INST_G_SHL_METADATA: GenericInstMetadata"));
    assert!(!output.contains("TiedOperandConstraint { result: 0, use_operand: 2 }"));
    assert!(output.contains("FixedUseConstraint { use_operand: 1, reg: REG_RCX }"));
    assert!(output.contains("pub fn generic_inst_metadata(opcode: veloc_lir::GenericOpcode)"));
}

#[test]
fn compile_stackslot_rules_generate_stackslot_operand_code() {
    let input = r#"
        (def-reg RSP (size 64) (hw-enc 4))
        (def-reg RBP (size 64) (hw-enc 5))

        (def-inst X86Load64Stack
          (operands (def $dst) (stackslot $slot))
          (encode
            (byte 0x8B)
            (byte (slot-base-hw-enc $slot))
            (imm32 (slot-offset $slot))))

        (def-inst X86Store64Stack
          (operands (use $src) (stackslot $slot))
          (encode
            (byte 0x89)
            (byte (slot-base-hw-enc $slot))
            (imm32 (slot-offset $slot))))

        (select-rule
          (match (schema StackLoad G_STACK_LOAD (dst (GPR64 $dst)) (slot $slot) @n))
          (emit (X86Load64Stack $dst $slot))
          (covers (@n))
          (cost 1))

        (select-rule
          (match (schema StackStore G_STACK_STORE (src (GPR64 $src)) (slot $slot) @n))
          (emit (X86Store64Stack $src $slot))
          (covers (@n))
          (cost 1))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("InstField::StackSlot"));
    assert!(output.contains("stack_frame.slots[slot]"));
    assert!(output.contains("TargetInst::X86Load64Stack"));
    assert!(output.contains("TargetInst::X86Store64Stack"));
}

#[test]
fn parse_select_rule_with_node_bind_and_covers() {
    let input = r#"
        (select-rule
          (match (G_ADD (GPR64 $x) (GPR64 $y) @n))
          (emit (X86Add64 $x $y))
          (covers (@n))
          (cost 1))
    "#;

    let module = parse(input).expect("parse should succeed");
    assert_eq!(module.defs.len(), 1);

    let Def::SelectRule(rule) = &module.defs[0] else {
        panic!("expected select-rule");
    };

    assert_eq!(rule.attrs.covers, vec!["n"]);
    assert_eq!(rule.attrs.cost, Some(1));
    assert_eq!(rule.patterns.len(), 1);
    assert!(matches!(rule.patterns[0], Pattern::NodeBind { .. }));
}

#[test]
fn parse_combine_rule_with_match_pair() {
    let input = r#"
        (combine-rule
          (match-pair ((G_SDIV $lhs $rhs @q)
                       (G_SREM $lhs $rhs @r)))
          (when ((same_block @q @r)))
          (replace (G_SDIVREM $lhs $rhs))
          (covers (@q @r))
          (cost 1))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::CombineRule(rule) = &module.defs[0] else {
        panic!("expected combine-rule");
    };

    assert_eq!(rule.match_kind, MatchKind::Pair);
    assert_eq!(rule.patterns.len(), 2);
    assert_eq!(rule.attrs.covers, vec!["q", "r"]);
}

#[test]
fn parse_rewrite_rule_definition() {
    let input = r#"
        (rewrite-rule
          (match (G_ADD (GPR64 $x) (GPR64 $y) @n))
          (replace (G_ADD (GPR64 $y) (GPR64 $x)))
          (cost 1)
          (priority 10))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::RewriteRule(rule) = &module.defs[0] else {
        panic!("expected rewrite-rule");
    };

    assert_eq!(rule.attrs.cost, Some(1));
    assert_eq!(rule.attrs.priority, Some(10));
    assert_eq!(rule.patterns.len(), 1);
}

#[test]
fn parse_pseudo_inst_definition() {
    let input = r#"
        (def-pseudo-inst X86SDivRem64Pseudo
          (operands (use $lhs) (use $rhs) (def $q) (def $r))
          (implicit-uses RAX)
          (implicit-defs RAX RDX EFLAGS)
          (clobbers EFLAGS))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::PseudoInst(inst) = &module.defs[0] else {
        panic!("expected def-pseudo-inst");
    };

    assert_eq!(inst.name, "X86SDivRem64Pseudo");
    assert_eq!(inst.operands.len(), 4);
    assert_eq!(inst.implicit_defs, vec!["RAX", "RDX", "EFLAGS"]);
}

#[test]
fn parse_block_operand_and_rel32_emit() {
    let input = r#"
        (def-inst X86Jmp
          (operands (block $target))
          (encode
            (byte 0xE9)
            (rel32 $target)))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::Inst(inst) = &module.defs[0] else {
        panic!("expected def-inst");
    };

    assert_eq!(inst.operands.len(), 1);
    assert!(matches!(
        inst.operands[0],
        veloc_isle::target::OperandConstraint::Block(ref name) if name == "target"
    ));
    assert!(matches!(
        inst.emit.get(1),
        Some(veloc_isle::target::EmitExpr::Rel32(name)) if name == "target"
    ));
}

#[test]
fn parse_def_abi_descriptor() {
    let input = r#"
        (def-abi X86_64SystemV
          (arch X86_64)
          (stack
            (align 16)
            (incoming-base RBP 16)
            (outgoing-slot 8 8))
          (args
            (class Integer (regs RDI RSI RDX RCX R8 R9)))
          (returns
            (class Integer (regs RAX RDX)))
          (preserved
            (gpr RBX RBP R12 R13 R14 R15))
          (classifier x86_64_sysv_classifier))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::Abi(abi) = &module.defs[0] else {
        panic!("expected def-abi");
    };

    assert_eq!(abi.name, "X86_64SystemV");
    assert_eq!(abi.arch, "X86_64");
    assert_eq!(abi.stack.align, Some(16));
    assert_eq!(abi.stack.incoming_base, Some(("RBP".to_string(), 16)));
    assert_eq!(
        abi.args[0].regs,
        vec!["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
    );
    assert_eq!(abi.returns[0].regs, vec!["RAX", "RDX"]);
    assert_eq!(abi.classifier.as_deref(), Some("x86_64_sysv_classifier"));
}

#[test]
fn parse_def_reg_with_reserved_and_role() {
    let input = r#"
        (def-reg RSP (size 64) (hw-enc 4) (reserved) (role stack-pointer))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::Reg(reg) = &module.defs[0] else {
        panic!("expected def-reg");
    };

    assert_eq!(reg.name, "RSP");
    assert_eq!(reg.size, 64);
    assert_eq!(reg.hw_enc, 4);
    assert!(reg.reserved);
    assert_eq!(reg.roles, vec!["stack-pointer"]);
}

#[test]
fn parse_fixed_use_operand_constraint() {
    let input = r#"
        (def-inst X86Shl32Cl
          (operands (def (tied $dst $src1)) (use (fixed RCX $count)))
          (clobbers EFLAGS))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::Inst(inst) = &module.defs[0] else {
        panic!("expected def-inst");
    };

    assert!(matches!(
        inst.operands[1],
        veloc_isle::target::OperandConstraint::FixedUse { ref reg, ref src }
            if reg == "RCX" && src == "count"
    ));
}

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

#[test]
fn test_error_reporting() {
    let input = "(select-rule (match (G_ADD $x $y @n)) (covers (@n)))";
    let res = veloc_isle::target::compile(input, "x86_64");
    if let Err(e) = res {
        println!("Expected error message:\n{}", e);
    } else {
        panic!("Should have failed to compile");
    }
}
