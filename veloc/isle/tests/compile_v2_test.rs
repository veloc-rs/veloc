use veloc_isle::compile;

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
    assert!(output.contains("MachineOperand::Block(target)"));
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

    assert!(output.contains("pub static ABI_X86_64SystemV: AbiDescriptor"));
    assert!(output.contains("AbiStackDescriptor"));
    assert!(output.contains("classifier: Some(\"x86_64_sysv_classifier\")"));
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
    assert!(output.contains("TargetTiedOperandMetadata { operand: 0 }"));
    assert!(output.contains("FixedUseConstraint { use_operand: 1, reg: REG_RCX }"));
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
          (emit (X86Add32 $y))
          (covers (@n))
          (cost 1))

        (select-rule
          (match (schema BinaryReg G_SHL (dst (GPR32 $dst)) (lhs (GPR32 $x)) (rhs (GPR32 $y)) @n))
          (emit (X86Shl32Cl $y))
          (covers (@n))
          (cost 1))
    "#;

    let output = compile(input, "x86_64").expect("compile should succeed");

    assert!(output.contains("pub const GENERIC_INST_G_ADD_METADATA: GenericInstMetadata"));
    assert!(output.contains("pub const GENERIC_INST_G_SHL_METADATA: GenericInstMetadata"));
    assert!(output.contains("TiedOperandConstraint { def_operand: 0, use_operand: 1 }"));
    assert!(output.contains("commute_operand_pairs: &[(1, 2)]"));
    assert!(output.contains("FixedUseConstraint { use_operand: 2, reg: REG_RCX }"));
    assert!(output.contains("pub fn generic_inst_metadata(opcode: crate::mir::GenericOpcode)"));
}
