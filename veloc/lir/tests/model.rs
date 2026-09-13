//! Construct, decode, validate, and interpret the standalone LIR model.
use veloc_lir::stages::{LegalizedLir, RawLir};
use veloc_lir::{
    ControlFlow, GenericOpcode, MachineFunction, MachineModule, MachineOperand, Reg, RegisterBank,
    SymbolTable, Type, TypeError, Writable,
};
use veloc_mir::Linkage;

#[test]
fn references_follow_all_store_edits_and_edge_arguments() {
    use veloc_lir::{BranchCondInfo, InstExtra, RefLocation, RefRole, VReg};
    let mut f = MachineFunction::<RawLir>::new("references".into());
    let a = f.alloc_vreg(Type::I64);
    let b = f.alloc_vreg(Type::I64);
    let dst = f.alloc_vreg(Type::I64);
    let block = f.create_synthetic_block();
    let add = f.writer().add(Writable(dst), a, a);
    let branch = f.writer().brcond(a, block, block);
    f.set_inst_extra(
        branch,
        InstExtra::BranchCond(BranchCondInfo {
            then_args: smallvec::smallvec![a, b],
            else_args: smallvec::smallvec![a],
        }),
    );
    assert_eq!(f.uses(a).count(), 5);
    assert!(f.uses(a).single().is_none());
    assert_eq!(f.defs(dst).single().unwrap().inst(), add);
    assert_eq!(f.inst(branch).uses().collect::<Vec<_>>(), [a, a, b, a]);
    let mut edges: Vec<_> = f
        .uses(a)
        .filter_map(|site| match site.location() {
            RefLocation::EdgeArg(index) => Some(index),
            _ => None,
        })
        .collect();
    edges.sort();
    assert_eq!(edges, [0, 2]);
    f.check_refs().unwrap();

    f.set_inst_operand(add, 1, MachineOperand::Use(b));
    assert_eq!(f.uses(a).count(), 4);
    f.replace_uses(VReg::from_u32(a.index()), VReg::from_u32(b.index()));
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.uses(b).count(), 6);
    for site in f.uses(b) {
        assert_eq!(site.reg(), b);
        assert_eq!(site.role(), RefRole::Use);
    }
    f.check_refs().unwrap();
    let clone = f.clone();
    f.clear_inst_extra(branch);
    assert_eq!(f.uses(b).count(), 3);
    assert_eq!(clone.uses(b).count(), 6);
    clone.check_refs().unwrap();

    // Replacing a tied input never renames its independent output.
    let rw = f
        .writer()
        .binary(veloc_lir::MachineOpcode::Target(0), Writable(a), b, a);
    f.replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.inst(rw).defs().collect::<Vec<_>>(), [a]);
    f.invalidate_inst(rw);
    f.check_refs().unwrap();

    let replacement = f.writer().copy(Writable(dst), a);
    f.replace_inst(add, replacement);
    assert!(f.inst(replacement).is_invalid());
    assert_eq!(f.uses(a).single().unwrap().inst(), add);
    assert_eq!(f.defs(dst).single().unwrap().inst(), add);
    f.check_refs().unwrap();

    // Both role changes and pooled-range reuse must unlink obsolete entries.
    for index in 0..128 {
        f.rewriter(add).constant(Writable(dst), index);
        assert_eq!(f.uses(a).count(), 0);
        f.rewriter(add).add(Writable(dst), a, a);
        assert_eq!(f.uses(a).count(), 2);
        f.check_refs().unwrap();
    }
    f.invalidate_inst(add);
    f.invalidate_inst(branch);
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.uses(b).count(), 0);
    assert_eq!(f.defs(dst).count(), 0);
    f.check_refs().unwrap();
}

#[test]
fn standalone_module_supports_instruction_and_stage_apis() {
    let mut function = MachineFunction::<RawLir>::new("example".into());
    let block = function.create_synthetic_block();
    let reg = function.alloc_vreg(Type::I64);
    let inst = function.writer().constant(Writable(reg), 42);
    function.append_inst_id_to_block(function.find_block_index(block).unwrap(), inst);
    let veloc_lir::InstView::Constant(constant) = function.inst(inst).generic_view().unwrap()
    else {
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
fn operand_edits_preserve_payload_but_replacement_discards_it() {
    use veloc_lir::{BranchInfo, InstExtra};
    let mut function = MachineFunction::<RawLir>::new("edit".into());
    let block = function.create_synthetic_block();
    let id = function.writer().br(block);
    let extra = InstExtra::Branch(BranchInfo {
        args: Default::default(),
    });
    function.set_inst_extra(id, extra.clone());

    let operands = function.inst(id).operands().to_vec();
    function.set_inst_operands(id, operands);
    assert_eq!(function.inst_extra(id), Some(&extra));

    function.invalidate_inst(id);
    assert_eq!(function.inst_extra(id), None);
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
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let inst = function
        .writer()
        .constant(Writable(veloc_lir::Reg::new_vreg(0)), 42);
    {
        let mut operands = function.inst(inst).operands().to_vec();
        operands.pop();
        function.set_inst_operands(inst, operands);
    }
    let error: veloc_lir::DecodeError = function.inst(inst).generic_view().unwrap_err();
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
    let mut function = MachineFunction::<RawLir>::new("test".into());
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
    let inst = function.writer().add(
        Writable(Reg::new_vreg(0)),
        Reg::new_vreg(1),
        Reg::new_vreg(2),
    );
    assert!(matches!(
        function.inst(inst).generic_view(),
        Ok(veloc_lir::InstView::BinaryReg(_))
    ));
}

#[test]
fn generated_builders_and_decoders_agree() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let lhs = Reg::new_vreg(1);
    let rhs = Reg::new_vreg(2);
    let veloc_lir::InstView::BinaryReg(decoded) = ({
        let id = function.writer().add(dst, lhs, rhs);
        function.inst(id).generic_view().unwrap()
    }) else {
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
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let flag = Writable(Reg::new_vreg(1));
    let lhs = Reg::new_vreg(2);
    let rhs = Reg::new_vreg(3);
    let carry = Reg::new_vreg(4);
    let add = function.writer().uaddo(dst, flag, lhs, rhs);
    let adc = function.writer().uadde(dst, flag, lhs, rhs, carry);
    let veloc_lir::InstView::BinaryRegWithFlags(add_view) =
        function.inst(add).generic_view().unwrap()
    else {
        panic!("expected flags");
    };
    let veloc_lir::InstView::BinaryRegWithFlags(adc_view) =
        function.inst(adc).generic_view().unwrap()
    else {
        panic!("expected flags");
    };
    assert_eq!(add_view.carry_in, None);
    assert_eq!(adc_view.carry_in, Some(carry));
    assert_eq!(add_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::UADDO);
    assert_eq!(adc_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::UADDE);
    {
        let mut operands = function.inst(add).operands().to_vec();
        operands.push(MachineOperand::Use(carry));
        function.set_inst_operands(add, operands);
    }
    {
        let mut operands = function.inst(adc).operands().to_vec();
        operands.pop();
        function.set_inst_operands(adc, operands);
    }
    assert!(function.inst(add).generic_view().is_err());
    assert!(function.inst(adc).generic_view().is_err());
}

#[test]
fn explicit_tied_mapping_preserves_input_and_output_register_identity() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let updated = Writable(Reg::new_vreg(1));
    let base = Reg::new_vreg(2);
    let inst = function.writer().indexed_load(dst, updated, base, 16);
    let veloc_lir::InstView::IndexedLoad(decoded) = function.inst(inst).generic_view().unwrap()
    else {
        panic!("expected IndexedLoad");
    };
    assert_eq!(
        (decoded.dst, decoded.wb_dst, decoded.base, decoded.offset),
        (dst.to_reg(), updated.to_reg(), base, 16)
    );
    assert!(
        matches!(function.inst(inst).operands()[1], MachineOperand::Def(reg) if reg == updated)
    );
}

#[test]
fn variable_views_preserve_call_and_return_operands() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    use veloc_lir::{CallCallee, InstView, SymbolId};
    let results: Vec<_> = (0..8).map(Reg::new_vreg).collect();
    let args: Vec<_> = (8..24).map(Reg::new_vreg).collect();
    let symbol = SymbolId::from_u32(3);
    let direct = function.writer().call(
        results.iter().copied().map(Writable),
        symbol,
        args.iter().copied(),
    );
    let indirect = function.writer().call_indirect(
        results.iter().copied().map(Writable),
        Reg::new_vreg(25),
        args.iter().copied(),
    );
    for (inst, callee) in [
        (direct, CallCallee::Direct(symbol)),
        (indirect, CallCallee::Indirect(Reg::new_vreg(25))),
    ] {
        let InstView::Call(call) = function.inst(inst).generic_view().unwrap() else {
            panic!("expected call");
        };
        assert_eq!(call.shape.callee, callee);
        assert_eq!(call.shape.defs.iter().collect::<Vec<_>>(), results);
        assert_eq!(call.shape.args.iter().collect::<Vec<_>>(), args);
    }
    let ret = function.writer().ret(args.iter().copied().collect());
    let InstView::Return(view) = function.inst(ret).generic_view().unwrap() else {
        panic!("expected return");
    };
    assert_eq!(view.values.len(), args.len());
    assert_eq!(view.values.iter().collect::<Vec<_>>(), args);
    let empty = function.writer().ret(Default::default());
    let InstView::Return(view) = function.inst(empty).generic_view().unwrap() else {
        panic!("expected return");
    };
    assert!(view.values.is_empty());
}

#[test]
fn views_reject_malformed_storage_without_semantic_validation() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    use veloc_lir::{CondCode, MachineOpcode, SymbolId};
    use veloc_mir::{FloatCC, IntCC};
    let dst = Writable(Reg::new_vreg(0));
    let src = Reg::new_vreg(1);
    let cmp = function.writer().icmp(dst, src, src, IntCC::Eq);
    function.set_inst_operand(
        cmp,
        3,
        MachineOperand::CondCode(CondCode::Float(FloatCC::Eq)),
    );
    assert!(function.inst(cmp).generic_view().is_err());
    let call = function.writer().call([dst], SymbolId::from_u32(0), [src]);
    {
        let mut operands = function.inst(call).operands().to_vec();
        operands.push(MachineOperand::Imm(0));
        function.set_inst_operands(call, operands);
    }
    assert!(function.inst(call).generic_view().is_err());
    let missing_callee = function.writer().call_indirect([dst], src, []);
    {
        let mut operands = function.inst(missing_callee).operands().to_vec();
        operands.pop();
        function.set_inst_operands(missing_callee, operands);
    }
    assert!(function.inst(missing_callee).generic_view().is_err());
    let ret = function.writer().ret(Default::default());
    {
        let mut operands = function.inst(ret).operands().to_vec();
        operands.push(MachineOperand::Def(dst));
        function.set_inst_operands(ret, operands);
    }
    assert!(function.inst(ret).generic_view().is_err());
    let target = function.writer().unary(MachineOpcode::Target(0), dst, src);
    assert!(function.inst(target).generic_view().is_err());
    function.invalidate_inst(target);
    assert!(function.inst(target).generic_view().is_err());
}
