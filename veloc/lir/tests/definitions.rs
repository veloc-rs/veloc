use veloc_lir::{ControlFlow, GenericOpcode, MachineInst, MachineOperand, Reg, Writable};
use veloc_lir::{Type, TypeError};

#[allow(dead_code)]
mod offline {
    use veloc_lir::GenericOpcode as Opcode;
    use veloc_mir::IntCC;
    include!(concat!(env!("OUT_DIR"), "/semantics.rs"));
}

#[test]
fn offline_semantics_use_the_same_emitter_for_operand_storage() {
    use veloc_semantics::Sort;
    assert_eq!(offline::SPECS.len(), 7);
    for spec in offline::SPECS {
        for width in [8, 16, 32, 64] {
            let sort = Sort::bv(width).unwrap();
            let inputs = vec![sort; spec.program.arity()];
            spec.program.instantiate(&inputs, &[sort], &[]).unwrap();
        }
    }
}

#[test]
fn logical_type_validation_is_separate_from_construction() {
    assert!(
        GenericOpcode::G_ADD
            .validate_types(&[Type::I32, Type::I32], &[Type::I32])
            .is_ok()
    );
    assert!(matches!(
        GenericOpcode::G_ADD.validate_types(&[Type::I32, Type::I64], &[Type::I32]),
        Err(TypeError::Pattern { .. })
    ));
    assert!(
        GenericOpcode::G_ADD
            .validate_types(&[Type::F32, Type::F32], &[Type::F32])
            .is_err()
    );
    assert!(
        GenericOpcode::G_FADD
            .validate_types(&[Type::F32, Type::F32], &[Type::F32])
            .is_ok()
    );
    assert!(
        GenericOpcode::G_UADDE
            .validate_types(
                &[Type::I32, Type::I32, Type::BOOL],
                &[Type::I32, Type::BOOL]
            )
            .is_ok()
    );
    assert!(
        GenericOpcode::G_UADDE
            .validate_types(&[Type::I32, Type::I32], &[Type::I32, Type::BOOL])
            .is_err()
    );
    // Physical construction deliberately cannot inspect register types.
    let inst = MachineInst::build_add(
        Writable(Reg::new_vreg(0)),
        Reg::new_vreg(1),
        Reg::new_vreg(2),
    );
    assert!(inst.as_binary_reg().is_ok());
}

#[test]
fn generated_builders_and_decoders_agree() {
    let dst = Writable(Reg::new_vreg(0));
    let lhs = Reg::new_vreg(1);
    let rhs = Reg::new_vreg(2);
    let decoded = MachineInst::build_add(dst, lhs, rhs)
        .as_binary_reg()
        .unwrap();
    assert_eq!(
        (decoded.dst, decoded.lhs, decoded.rhs),
        (dst.to_reg(), lhs, rhs)
    );
    assert_eq!(GenericOpcode::G_ADD.control(), ControlFlow::Next);
    assert_eq!(GenericOpcode::G_BRCOND.control(), ControlFlow::Jump);
    assert_eq!(GenericOpcode::G_UNREACHABLE.control(), ControlFlow::Trap);
}

#[test]
fn carry_input_is_required_exactly_for_carry_instructions() {
    let dst = Writable(Reg::new_vreg(0));
    let flag = Writable(Reg::new_vreg(1));
    let lhs = Reg::new_vreg(2);
    let rhs = Reg::new_vreg(3);
    let carry = Reg::new_vreg(4);
    let mut add = MachineInst::build_uaddo(dst, flag, lhs, rhs);
    let mut adc = MachineInst::build_uadde(dst, flag, lhs, rhs, carry);
    assert_eq!(add.as_binary_reg_with_flags().unwrap().carry_in, None);
    assert_eq!(
        adc.as_binary_reg_with_flags().unwrap().carry_in,
        Some(carry)
    );
    add.operands.push(MachineOperand::Use(carry));
    adc.operands.pop();
    assert!(add.as_binary_reg_with_flags().is_err());
    assert!(adc.as_binary_reg_with_flags().is_err());
}
