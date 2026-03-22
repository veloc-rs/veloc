use veloc_isle::{parse, Def, MatchKind, Pattern};

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
          (replace (G_ADD (GPR64 $y) (GPR64 $x) @n))
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
        veloc_isle::OperandConstraint::Block(ref name) if name == "target"
    ));
    assert!(matches!(
        inst.emit.get(1),
        Some(veloc_isle::EmitExpr::Rel32(name)) if name == "target"
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
        veloc_isle::OperandConstraint::FixedUse { ref reg, ref src }
            if reg == "RCX" && src == "count"
    ));
}
