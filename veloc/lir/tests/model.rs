//! Construct, decode, validate, and interpret the standalone LIR model.
use veloc_lir::stages::{LegalizedLir, RawLir};
use veloc_lir::{
    ControlFlow, GenericOpcode, MachineFunction, MachineInst, MachineModule, MachineOperand, Reg,
    RegisterBank, SymbolTable, Type, TypeError, Writable,
};
use veloc_mir::Linkage;

#[test]
fn standalone_module_supports_instruction_and_stage_apis() {
    let mut function = MachineFunction::<RawLir>::new("example".into());
    let block = function.create_synthetic_block();
    let reg = function.alloc_vreg(Type::I64);
    let inst = function.alloc_inst(MachineInst::build_constant(Writable(reg), 42));
    function.append_inst_id_to_block(function.find_block_index(block).unwrap(), inst);
    let veloc_lir::InstView::Constant(constant) = function.dfg[inst].generic_view().unwrap() else {
        panic!("expected constant");
    };
    assert_eq!(constant.imm, 42);

    let mut module = MachineModule::new("standalone".into());
    let id = module.add_function(function);
    assert_eq!(module.find_function_by_name("example"), Some(id));
    assert_eq!(module.functions[id].block_insts(0), &[inst]);
    let mut function = module.functions[id].clone().into_stage::<LegalizedLir>();
    let banked = function.alloc_vreg_in_bank(Type::I64, RegisterBank::GPR);
    assert!(banked.is_vreg());
}

#[test]
fn symbol_interning_does_not_require_a_source_module() {
    let mut symbols = SymbolTable::new();
    let first = symbols.get_or_create_function("callee", Linkage::Import);
    assert_eq!(
        symbols.get_or_create_function("callee", Linkage::Import),
        first
    );
    assert_eq!(symbols.get(first).linkage, Linkage::Import);
    assert_ne!(
        symbols.get_or_create_function("other", Linkage::Export),
        first
    );
}

#[test]
fn decode_errors_are_owned_by_lir() {
    let mut inst = MachineInst::build_constant(Writable(veloc_lir::Reg::new_vreg(0)), 42);
    inst.operands.pop();
    let error: veloc_lir::DecodeError = inst.generic_view().unwrap_err();
    assert!(matches!(
        error.opcode,
        veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::G_CONSTANT)
    ));
    assert!(!error.reason.is_empty());
    assert!(error.to_string().contains("invalid LIR instruction"));
}

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
    assert!(matches!(
        inst.generic_view(),
        Ok(veloc_lir::InstView::BinaryReg(_))
    ));
}

#[test]
fn generated_builders_and_decoders_agree() {
    let dst = Writable(Reg::new_vreg(0));
    let lhs = Reg::new_vreg(1);
    let rhs = Reg::new_vreg(2);
    let veloc_lir::InstView::BinaryReg(decoded) = MachineInst::build_add(dst, lhs, rhs)
        .generic_view()
        .unwrap()
    else {
        panic!("expected BinaryReg");
    };
    assert_eq!(
        (decoded.dst, decoded.lhs, decoded.rhs),
        (dst.to_reg(), lhs, rhs)
    );
    assert_eq!(decoded.opcode, veloc_lir::BinaryRegOpcode::ADD);
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
    let veloc_lir::InstView::BinaryRegWithFlags(add_view) = add.generic_view().unwrap() else {
        panic!("expected flags");
    };
    let veloc_lir::InstView::BinaryRegWithFlags(adc_view) = adc.generic_view().unwrap() else {
        panic!("expected flags");
    };
    assert_eq!(add_view.carry_in, None);
    assert_eq!(adc_view.carry_in, Some(carry));
    assert_eq!(add_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::UADDO);
    assert_eq!(adc_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::UADDE);
    add.operands.push(MachineOperand::Use(carry));
    adc.operands.pop();
    assert!(add.generic_view().is_err());
    assert!(adc.generic_view().is_err());
}

#[test]
fn explicit_tied_mapping_preserves_input_and_output_register_identity() {
    let dst = Writable(Reg::new_vreg(0));
    let updated = Writable(Reg::new_vreg(1));
    let base = Reg::new_vreg(2);
    let inst = MachineInst::build_indexed_load(dst, updated, base, 16);
    let veloc_lir::InstView::IndexedLoad(decoded) = inst.generic_view().unwrap() else {
        panic!("expected IndexedLoad");
    };
    assert_eq!(
        (decoded.dst, decoded.wb_dst, decoded.base, decoded.offset),
        (dst.to_reg(), updated.to_reg(), base, 16)
    );
    assert!(matches!(inst.operands[1], MachineOperand::TiedDefUse(reg) if reg == updated));
}

#[test]
fn variable_views_preserve_call_and_return_operands() {
    use veloc_lir::{CallCallee, InstView, SymbolId};
    let results: Vec<_> = (0..8).map(Reg::new_vreg).collect();
    let args: Vec<_> = (8..24).map(Reg::new_vreg).collect();
    let symbol = SymbolId::from_u32(3);
    let direct = MachineInst::build_call(
        results.iter().copied().map(Writable),
        symbol,
        args.iter().copied(),
    );
    let indirect = MachineInst::build_call_indirect(
        results.iter().copied().map(Writable),
        Reg::new_vreg(25),
        args.iter().copied(),
    );
    for (inst, callee) in [
        (&direct, CallCallee::Direct(symbol)),
        (&indirect, CallCallee::Indirect(Reg::new_vreg(25))),
    ] {
        let InstView::Call(call) = inst.generic_view().unwrap() else {
            panic!("expected call");
        };
        assert_eq!(call.shape.callee, callee);
        assert_eq!(call.shape.defs.iter().collect::<Vec<_>>(), results);
        assert_eq!(call.shape.args.iter().collect::<Vec<_>>(), args);
    }
    let ret = MachineInst::build_ret(args.iter().copied().collect());
    let InstView::Return(view) = ret.generic_view().unwrap() else {
        panic!("expected return");
    };
    assert_eq!(view.values.len(), args.len());
    assert_eq!(view.values.iter().collect::<Vec<_>>(), args);
    let empty = MachineInst::build_ret(Default::default());
    let InstView::Return(view) = empty.generic_view().unwrap() else {
        panic!("expected return");
    };
    assert!(view.values.is_empty());
}

#[test]
fn views_reject_malformed_storage_without_semantic_validation() {
    use veloc_lir::{CondCode, MachineOpcode, SymbolId};
    use veloc_mir::{FloatCC, IntCC};
    let dst = Writable(Reg::new_vreg(0));
    let src = Reg::new_vreg(1);
    let mut cmp = MachineInst::build_icmp(dst, src, src, IntCC::Eq);
    cmp.operands[3] = MachineOperand::CondCode(CondCode::Float(FloatCC::Eq));
    assert!(cmp.generic_view().is_err());
    let mut call = MachineInst::build_call([dst], SymbolId::from_u32(0), [src]);
    call.operands.push(MachineOperand::Imm(0));
    assert!(call.generic_view().is_err());
    let mut missing_callee = MachineInst::build_call_indirect([dst], src, []);
    missing_callee.operands.pop();
    assert!(missing_callee.generic_view().is_err());
    let mut ret = MachineInst::build_ret(Default::default());
    ret.operands.push(MachineOperand::Def(dst));
    assert!(ret.generic_view().is_err());
    let target = MachineInst::build_unary(MachineOpcode::Target(0), dst, src);
    assert!(target.generic_view().is_err());
    assert!(MachineInst::invalid().generic_view().is_err());
}
