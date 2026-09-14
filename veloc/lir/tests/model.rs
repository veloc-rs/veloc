//! Construct, decode, validate, and interpret the standalone LIR model.
use veloc_lir::stages::{LegalizedLir, RawLir};
use veloc_lir::{
    ControlFlow, GenericOpcode, InstField, MachineFunction, MachineModule, Reg, RegisterBank,
    SymbolTable, Type, TypeError, Writable,
};
use veloc_lir::{InstBuild, InstRead};
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

    f.set_inst_input(add, 1, b);
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
    // Register references and attribute edits address independent storage domains.
    let mixed = f.writer().write(
        veloc_lir::MachineOpcode::Target(0),
        &[],
        &[a, a],
        &[InstField::Imm(7), InstField::Imm(9)],
    );
    assert_eq!(f.inst(mixed).inputs(), &[a, a]);
    assert_eq!(f.inst(mixed).fields().len(), 2);
    f.set_inst_input(mixed, 1, b);
    assert_eq!(f.inst(mixed).inputs()[1], b);
    f.check_refs().unwrap();
    f.replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
    assert_eq!(f.inst(mixed).inputs(), &[b, b]);
    // Attribute edits do not touch register references or input storage.
    let inputs = f.inst(mixed).inputs().as_ptr();
    f.set_inst_field(mixed, 0, InstField::Imm(11));
    assert_eq!(f.inst(mixed).inputs().as_ptr(), inputs);
    assert_eq!(f.inst(mixed).inputs(), &[b, b]);
    f.check_refs().unwrap();
    f.invalidate_inst(mixed);
    // Result edits and implicit physical effects have independent locations.
    f.set_inst_result(add, 0, b);
    assert_eq!(f.defs(dst).count(), 0);
    assert_eq!(
        f.defs(b).single().unwrap().location(),
        RefLocation::Result(0)
    );
    let preg = Reg::new_preg(3);
    f.set_inst_effects(
        add,
        veloc_lir::RegEffects {
            uses: vec![preg],
            defs: vec![preg],
        },
    );
    assert_eq!(
        f.uses(preg).single().unwrap().location(),
        RefLocation::ImplicitUse(0)
    );
    assert_eq!(
        f.defs(preg).single().unwrap().location(),
        RefLocation::ImplicitDef(0)
    );
    assert_eq!(f.inst(add).results(), &[b]);
    f.check_refs().unwrap();
    f.invalidate_inst(add);
    assert_eq!(f.uses(preg).count(), 0);
    assert_eq!(f.defs(preg).count(), 0);
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
    let veloc_lir::InstView::Constant(constant) = function.inst(inst).view() else {
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

    let operands = function.inst(id).fields().to_vec();
    function.set_inst_fields(id, &operands);
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
fn validation_errors_are_owned_by_lir() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let inst = function
        .writer()
        .constant(Writable(veloc_lir::Reg::new_vreg(0)), 42);
    {
        let mut operands = function.inst(inst).fields().to_vec();
        operands.pop();
        function.set_inst_fields(inst, &operands);
    }
    let error: veloc_lir::ValidationError = function.inst(inst).validate().unwrap_err();
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
        function.inst(inst).view(),
        veloc_lir::InstView::BinaryReg(_)
    ));
}

#[test]
fn generated_builders_and_views_agree() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let lhs = Reg::new_vreg(1);
    let rhs = Reg::new_vreg(2);
    let veloc_lir::InstView::BinaryReg(decoded) = ({
        let id = function.writer().add(dst, lhs, rhs);
        function.inst(id).view()
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
    let veloc_lir::InstView::BinaryRegWithFlags(add_view) = function.inst(add).view() else {
        panic!("expected flags");
    };
    let veloc_lir::InstView::BinaryRegWithFlags(adc_view) = function.inst(adc).view() else {
        panic!("expected flags");
    };
    assert_eq!(add_view.carry_in, None);
    assert_eq!(adc_view.carry_in, Some(carry));
    assert_eq!(add_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::UADDO);
    assert_eq!(adc_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::UADDE);
    {
        let regs = [lhs, rhs, lhs];
        function.rewriter(add).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::G_UADDO),
            &[dst.to_reg(), flag.to_reg()],
            &regs,
            &[],
        );
    }
    {
        function.rewriter(adc).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::G_UADDE),
            &[dst.to_reg(), flag.to_reg()],
            &[lhs, rhs],
            &[],
        );
    }
    assert!(function.inst(add).validate().is_err());
    assert!(function.inst(adc).validate().is_err());
}

#[test]
fn explicit_tied_mapping_preserves_input_and_output_register_identity() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let updated = Writable(Reg::new_vreg(1));
    let base = Reg::new_vreg(2);
    let inst = function.writer().indexed_load(dst, updated, base, 16);
    let veloc_lir::InstView::IndexedLoad(decoded) = function.inst(inst).view() else {
        panic!("expected IndexedLoad");
    };
    assert_eq!(
        (decoded.dst, decoded.wb_dst, decoded.base, decoded.offset),
        (dst.to_reg(), updated.to_reg(), base, 16)
    );
    assert_eq!(function.inst(inst).results()[1], updated.to_reg());
}

#[test]
fn variable_views_preserve_call_and_return_operands() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    use veloc_lir::{InstView, SymbolId};
    let results: Vec<_> = (0..8).map(Reg::new_vreg).collect();
    let args: Vec<_> = (8..24).map(Reg::new_vreg).collect();
    let symbol = SymbolId::from_u32(3);
    let direct = function.writer().call(&results, symbol, &args);
    let indirect = function
        .writer()
        .callind(&results, Reg::new_vreg(25), &args);
    for inst in [direct, indirect] {
        let (actual_results, actual_args) = match function.inst(inst).view() {
            InstView::Call(call) => {
                assert_eq!(call.callee, symbol);
                (call.results, call.args)
            }
            InstView::CallIndirect(call) => {
                assert_eq!(call.callee, Reg::new_vreg(25));
                (call.results, call.args)
            }
            _ => panic!("expected call"),
        };
        assert_eq!(actual_results, results);
        assert_eq!(actual_args, args);
        function.inst(inst).validate().unwrap();
    }
    let ret = function.writer().ret(&args);
    let InstView::Return(view) = function.inst(ret).view() else {
        panic!("expected return");
    };
    assert_eq!(view.values.len(), args.len());
    assert_eq!(view.values.iter().copied().collect::<Vec<_>>(), args);
    let empty = function.writer().ret(&[]);
    let InstView::Return(view) = function.inst(empty).view() else {
        panic!("expected return");
    };
    assert!(view.values.is_empty());
}

#[test]
fn optional_validation_is_separate_from_direct_views() {
    let mut function = MachineFunction::<RawLir>::new("test".into());
    use veloc_lir::{MachineOpcode, SymbolId};
    use veloc_mir::{FloatCC, IntCC};
    let dst = Writable(Reg::new_vreg(0));
    let src = Reg::new_vreg(1);
    // Property constraints are defs-driven and remain opt-in.
    let arg = function.writer().arg(dst, -1);
    assert!(matches!(
        function.inst(arg).view(),
        veloc_lir::InstView::Arg(_)
    ));
    assert!(function.inst(arg).validate().is_err());
    function.set_inst_field(arg, 0, InstField::Imm(0));
    function.inst(arg).validate().unwrap();

    let cmp = function.writer().icmp(dst, src, src, IntCC::Eq);
    function.set_inst_field(cmp, 0, InstField::FloatCC(FloatCC::Eq));
    assert!(function.inst(cmp).validate().is_err());
    let call = function
        .writer()
        .call(&[dst.to_reg()], SymbolId::from_u32(0), &[src]);
    {
        let mut operands = function.inst(call).fields().to_vec();
        operands.push(InstField::Imm(0));
        function.set_inst_fields(call, &operands);
    }
    assert!(function.inst(call).validate().is_err());
    let missing_callee = function.writer().callind(&[dst.to_reg()], src, &[]);
    {
        function.rewriter(missing_callee).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::G_CALLIND),
            &[dst.to_reg()],
            &[],
            &[],
        );
    }
    assert!(function.inst(missing_callee).validate().is_err());
    let ret = function.writer().ret(&[]);
    {
        function.set_inst_results(ret, &[dst.to_reg()]);
    }
    assert!(function.inst(ret).validate().is_err());
    let target = function.writer().unary(MachineOpcode::Target(0), dst, src);
    assert!(function.inst(target).validate().is_err());
    function.invalidate_inst(target);
    assert!(function.inst(target).validate().is_err());

    // Access does not run the optional full shape check: unrelated extra
    // attributes are rejected by validation, not by reading an add's registers.
    let add = function.writer().add(dst, src, src);
    function.set_inst_fields(add, &[InstField::Imm(7)]);
    assert!(function.inst(add).validate().is_err());
    assert!(matches!(
        function.inst(add).view(),
        veloc_lir::InstView::BinaryReg(_)
    ));

    // A wrong field variant is an internal invariant failure, not Result flow.
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| function.inst(cmp).view()))
            .is_err()
    );
}
