//! Construct, decode, validate, and interpret the standalone LIR model.
use veloc_lir::{
    ControlFlow, GenericOpcode, InstField, MachineFunction, MachineModule, Reg, RegisterBank,
    SymbolTable, Type, TypeError, Writable,
};
use veloc_lir::{InstBuild, InstRead};
use veloc_mir::Linkage;

#[test]
fn function_editor_preserves_layout_and_references() {
    let mut f = MachineFunction::new("layout".into());
    let entry = f.editor().create_block();
    let exit = f.editor().create_block();
    let a = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(1), &[], &[], &[]);
    let b = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(2), &[], &[], &[]);
    f.editor().append_inst(veloc_lir::BlockId::from_u32(0), a);
    f.editor().append_inst(veloc_lir::BlockId::from_u32(0), b);
    f.editor().move_block_before(exit, entry);
    assert_eq!(f.entry_block(), Some(entry));
    assert_eq!(f.blocks().collect::<Vec<_>>(), [exit, entry]);

    assert_eq!(f.inst_block(a), Some(entry));
    f.editor().reorder_block(entry, &[b, a]);
    for invalid in [&[a][..], &[a, a][..]] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f.editor().reorder_block(entry, invalid);
            }))
            .is_err()
        );
        assert_eq!(f.block_insts(entry).collect::<Vec<_>>(), [b, a]);
    }
    let (result, _) = f.track_edits(|f| {
        f.editor().invalidate_inst(b);
        Err::<(), _>("stop")
    });
    assert_eq!(result, Err("stop"));
    assert_eq!(f.block_insts(entry).collect::<Vec<_>>(), &[a]);
    assert_eq!(f.inst_block(b), None);
    assert_eq!(f.inst_block(a), Some(entry));
    assert!(f.inst(b).is_invalid());
    f.check_refs().unwrap();

    let c = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(3), &[], &[], &[]);
    let d = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(4), &[], &[], &[]);
    let (tail, changes) = f.track_edits(|f| {
        f.editor().insert_after(a, c);
        f.editor().insert_before(c, d);
        let tail = f.editor().split_block(d);
        f.editor().move_before(c, a);
        tail
    });
    assert!(changes.blocks.contains(&entry));
    assert!(changes.blocks.contains(&tail));
    assert_eq!(f.block_insts(entry).collect::<Vec<_>>(), [c, a]);
    assert_eq!(f.block_insts(entry).rev().collect::<Vec<_>>(), [a, c]);
    assert_eq!(f.block_insts(tail).collect::<Vec<_>>(), [d]);
    assert_eq!(f.layout().prev_inst(a), Some(c));
    assert_eq!(f.inst_block(d), Some(tail));
    f.editor().erase_block(tail);
    assert!(f.inst(d).is_invalid());
    assert_eq!(f.inst_block(d), None);
    assert!(f.block_params(tail).is_none());
    let new = f.editor().create_block();
    assert_ne!(new, tail, "erased block identities must not be recycled");

    let x = f.editor().alloc_vreg(Type::I64);
    let y = f.editor().alloc_vreg(Type::I64);
    let branch = f.editor().writer().br(entry);
    f.editor().append_inst(exit, branch);
    let (_, changes) = f.track_edits(|f| f.editor().redirect_edge(branch, 0, new, &[x]));
    assert!(changes.insts.contains(&branch));
    assert_eq!(f.uses(x).count(), 1);
    f.editor().redirect_edge(branch, 0, entry, &[y]);
    assert_eq!(f.uses(x).count(), 0);
    assert_eq!(f.uses(y).count(), 1);
    let cloned = f.clone();
    f.editor().invalidate_inst(branch);
    assert_eq!(f.inst_block(branch), None);
    assert_eq!(f.uses(y).count(), 0);
    assert_eq!(cloned.inst_block(branch), Some(exit));
    assert_eq!(cloned.uses(y).count(), 1);
    f.check_refs().unwrap();
    cloned.check_refs().unwrap();
    // Tracking is scoped even when an editor callback panics.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f.track_edits(|_| panic!("abandon edit"));
    }));
    let (_, changes) = f.track_edits(|f| f.editor().move_before(a, c));
    assert!(changes.blocks.contains(&entry));

    // Replacements update the real layout immediately, including retained roots.
    let mut edit = f.editor();
    let replacement = edit
        .writer()
        .write(veloc_lir::MachineOpcode::Target(5), &[], &[], &[]);
    edit.replace_with(a, &[replacement, a]);
    assert_eq!(
        edit.block_insts(entry).collect::<Vec<_>>(),
        [replacement, a, c]
    );
    edit.replace_with(a, &[]);
    assert!(edit.inst(a).is_invalid());
    assert_eq!(
        edit.block_insts(entry).collect::<Vec<_>>(),
        [replacement, c]
    );
    let next = edit
        .writer()
        .write(veloc_lir::MachineOpcode::Target(6), &[], &[], &[]);
    edit.replace_with(c, &[next]);
    assert!(edit.inst(c).is_invalid());
    assert_eq!(edit.inst_block(next), Some(entry));
    assert_eq!(
        edit.block_insts(entry).collect::<Vec<_>>(),
        [replacement, next]
    );
    edit.check_refs().unwrap();
}

#[test]
fn references_follow_all_store_edits_and_edge_arguments() {
    use veloc_lir::{BranchCondInfo, InstExtra, RefRole, VReg};
    let mut f = MachineFunction::new("references".into());
    let a = f.editor().alloc_vreg(Type::I64);
    let b = f.editor().alloc_vreg(Type::I64);
    let dst = f.editor().alloc_vreg(Type::I64);
    let block = f.editor().create_block();
    let add = f.editor().writer().add(Writable(dst), a, a);
    let branch = f.editor().writer().brcond(a, block, block);
    f.editor().set_inst_extra(
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
    // Every occurrence has its own slot, including repeated edge arguments.
    let slots: std::collections::HashSet<_> = f.uses(a).map(|site| site.operand()).collect();
    assert_eq!(slots.len(), 5);
    assert!(slots.contains(&f.input_id(add, 0)));
    assert!(slots.contains(&f.input_id(add, 1)));
    f.check_refs().unwrap();

    let input_slots = [f.input_id(add, 0), f.input_id(add, 1)];
    let result_slot = f.result_id(add, 0);
    f.editor().set_inst_input(add, 1, b);
    assert_eq!([f.input_id(add, 0), f.input_id(add, 1)], input_slots);
    assert_eq!(f.result_id(add, 0), result_slot);
    assert_eq!(f.operand(input_slots[1]), b);
    assert_eq!(f.uses(a).count(), 4);
    f.editor()
        .replace_uses(VReg::from_u32(a.index()), VReg::from_u32(b.index()));
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.uses(b).count(), 6);
    for site in f.uses(b) {
        assert_eq!(site.reg(), b);
        assert_eq!(site.role(), RefRole::Use);
    }
    f.check_refs().unwrap();
    let clone = f.clone();
    f.editor().clear_inst_extra(branch);
    assert_eq!(f.uses(b).count(), 3);
    assert_eq!(clone.uses(b).count(), 6);
    clone.check_refs().unwrap();

    // Replacing a tied input never renames its independent output.
    let rw = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(0), &[a], &[b, a], &[]);
    f.editor()
        .replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.inst(rw).defs().collect::<Vec<_>>(), [a]);
    f.editor().invalidate_inst(rw);
    f.check_refs().unwrap();

    let replacement = f.editor().writer().copy(Writable(dst), a);
    f.editor().replace_inst(add, replacement);
    assert!(f.inst(replacement).is_invalid());
    assert_eq!(f.uses(a).single().unwrap().inst(), add);
    assert_eq!(f.defs(dst).single().unwrap().inst(), add);
    f.check_refs().unwrap();

    // Both role changes and pooled-range reuse must unlink obsolete entries.
    for index in 0..128 {
        f.editor().rewriter(add).constant(Writable(dst), index);
        assert_eq!(f.uses(a).count(), 0);
        f.editor().rewriter(add).add(Writable(dst), a, a);
        assert_eq!(f.uses(a).count(), 2);
        f.check_refs().unwrap();
    }
    // Register references and attribute edits address independent storage domains.
    let mixed = f.editor().writer().write(
        veloc_lir::MachineOpcode::Target(0),
        &[],
        &[a, a],
        &[InstField::Imm(7), InstField::Imm(9)],
    );
    assert_eq!(f.inst(mixed).inputs(), &[a, a]);
    assert_eq!(f.inst(mixed).fields().len(), 2);
    f.editor().set_inst_input(mixed, 1, b);
    assert_eq!(f.inst(mixed).inputs()[1], b);
    f.check_refs().unwrap();
    f.editor()
        .replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
    assert_eq!(f.inst(mixed).inputs(), &[b, b]);
    // Attribute edits do not touch register references or input storage.
    let inputs = f.inst(mixed).inputs().as_ptr();
    f.editor().set_inst_field(mixed, 0, InstField::Imm(11));
    assert_eq!(f.inst(mixed).inputs().as_ptr(), inputs);
    assert_eq!(f.inst(mixed).inputs(), &[b, b]);
    f.check_refs().unwrap();
    f.editor().invalidate_inst(mixed);
    // Result edits and implicit physical effects have independent locations.
    f.editor().set_inst_result(add, 0, b);
    assert_eq!(f.defs(dst).count(), 0);
    assert_eq!(f.defs(b).single().unwrap().operand(), f.result_id(add, 0));
    let preg = Reg::new_preg(3);
    f.editor().set_inst_effects(
        add,
        veloc_lir::RegEffects {
            uses: vec![preg],
            defs: vec![preg],
        },
    );
    assert_eq!(f.uses(preg).single().unwrap().role(), RefRole::Use);
    assert_eq!(f.defs(preg).single().unwrap().role(), RefRole::Def);
    assert_eq!(f.inst(add).results(), &[b]);
    f.check_refs().unwrap();
    f.editor().invalidate_inst(add);
    assert_eq!(f.uses(preg).count(), 0);
    assert_eq!(f.defs(preg).count(), 0);
    f.editor().invalidate_inst(branch);
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.uses(b).count(), 0);
    assert_eq!(f.defs(dst).count(), 0);
    f.check_refs().unwrap();
    // Jump-table edges and implicit effects use the same pool as explicit operands.
    let source = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(42), &[dst], &[a], &[]);
    f.editor().set_inst_extra(
        source,
        InstExtra::BrTable(veloc_lir::BrTableInfo {
            targets: vec![
                veloc_lir::BrTableTarget {
                    block,
                    args: smallvec::smallvec![a, a, b],
                },
                veloc_lir::BrTableTarget {
                    block,
                    args: smallvec::smallvec![],
                },
                veloc_lir::BrTableTarget {
                    block,
                    args: smallvec::smallvec![a],
                },
            ],
        }),
    );
    f.editor().set_inst_effects(
        source,
        veloc_lir::RegEffects {
            uses: vec![preg],
            defs: vec![preg],
        },
    );
    let source_slot = f.input_id(source, 0);
    let source_slots: std::collections::HashSet<_> = f.uses(a).map(|r| r.operand()).collect();
    f.editor()
        .replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
    assert!(source_slots.iter().all(|&slot| f.operand(slot) == b));
    let Some(veloc_lir::InstExtraRef::BrTable(table)) = f.inst_extra(source) else {
        panic!("missing table");
    };
    assert_eq!(
        table.targets().map(|t| t.args.to_vec()).collect::<Vec<_>>(),
        [vec![b, b, b], vec![], vec![b]]
    );
    f.editor().replace_inst(branch, source);
    assert_eq!(f.input_id(branch, 0), source_slot);
    assert!(f.uses(b).all(|r| r.inst() == branch));
    assert_eq!(f.uses(preg).single().unwrap().inst(), branch);
    assert_eq!(f.defs(preg).single().unwrap().inst(), branch);
    f.check_refs().unwrap();
    f.editor().invalidate_inst(branch);
    f.check_refs().unwrap();
    assert_eq!(f.uses(b).count(), 0);
    assert_eq!(f.defs(dst).count(), 0);
}

#[test]
fn standalone_module_supports_instruction_and_stage_apis() {
    let mut function = MachineFunction::new("example".into());
    let block = function.editor().create_block();
    let reg = function.editor().alloc_vreg(Type::I64);
    let inst = function.editor().writer().constant(Writable(reg), 42);
    function.editor().append_inst(block, inst);
    let veloc_lir::InstView::Constant(constant) = function.inst(inst).view() else {
        panic!("expected constant");
    };
    assert_eq!(constant.imm, 42);

    let mut module = MachineModule::new("standalone".into());
    let id = module.add_function(function);
    assert_eq!(module.find_function_by_name("example"), Some(id));
    assert_eq!(
        module.functions[id]
            .block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>(),
        &[inst]
    );
    let mut function = module.functions[id].clone();
    let banked = function
        .editor()
        .alloc_vreg_in_bank(Type::I64, RegisterBank::GPR);
    assert!(banked.is_vreg());
}

#[test]
fn operand_edits_preserve_payload_but_replacement_discards_it() {
    use veloc_lir::{BranchInfo, InstExtra};
    let mut function = MachineFunction::new("edit".into());
    let block = function.editor().create_block();
    let id = function.editor().writer().br(block);
    let extra = InstExtra::Branch(BranchInfo {
        args: Default::default(),
    });
    function.editor().set_inst_extra(id, extra.clone());

    let operands = function.inst(id).fields().to_vec();
    function.editor().set_inst_fields(id, &operands);
    assert_eq!(function.inst_extra(id).map(|e| e.to_owned()), Some(extra));

    function.editor().invalidate_inst(id);
    assert!(function.inst_extra(id).is_none());
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
    let mut function = MachineFunction::new("test".into());
    let inst = function
        .editor()
        .writer()
        .constant(Writable(veloc_lir::Reg::new_vreg(0)), 42);
    {
        let mut operands = function.inst(inst).fields().to_vec();
        operands.pop();
        function.editor().set_inst_fields(inst, &operands);
    }
    let error: veloc_lir::ValidationError = function.inst(inst).validate().unwrap_err();
    assert!(matches!(
        error.opcode,
        veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::Constant)
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
    let mut function = MachineFunction::new("test".into());
    assert!(
        GenericOpcode::Add
            .validate_types(&[Type::I32, Type::I32], &[Type::I32])
            .is_ok()
    );
    assert!(matches!(
        GenericOpcode::Add.validate_types(&[Type::I32, Type::I64], &[Type::I32]),
        Err(TypeError::Pattern { .. })
    ));
    assert!(
        GenericOpcode::Add
            .validate_types(&[Type::F32, Type::F32], &[Type::F32])
            .is_err()
    );
    assert!(
        GenericOpcode::Fadd
            .validate_types(&[Type::F32, Type::F32], &[Type::F32])
            .is_ok()
    );
    assert!(
        GenericOpcode::Uadde
            .validate_types(
                &[Type::I32, Type::I32, Type::BOOL],
                &[Type::I32, Type::BOOL]
            )
            .is_ok()
    );
    assert!(
        GenericOpcode::Uadde
            .validate_types(&[Type::I32, Type::I32], &[Type::I32, Type::BOOL])
            .is_err()
    );
    // Physical construction deliberately cannot inspect register types.
    let inst = function.editor().writer().add(
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
    let mut function = MachineFunction::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let lhs = Reg::new_vreg(1);
    let rhs = Reg::new_vreg(2);
    let veloc_lir::InstView::BinaryReg(decoded) = ({
        let id = function.editor().writer().add(dst, lhs, rhs);
        function.inst(id).view()
    }) else {
        panic!("expected BinaryReg");
    };
    assert_eq!(
        (decoded.dst, decoded.lhs, decoded.rhs),
        (dst.to_reg(), lhs, rhs)
    );
    assert_eq!(decoded.opcode, veloc_lir::BinaryRegOpcode::Add);
    assert_eq!(GenericOpcode::Add.control(), ControlFlow::Next);
    assert_eq!(GenericOpcode::Brcond.control(), ControlFlow::Jump);
    assert_eq!(GenericOpcode::Unreachable.control(), ControlFlow::Trap);
}

#[test]
fn carry_input_is_required_exactly_for_carry_instructions() {
    let mut function = MachineFunction::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let flag = Writable(Reg::new_vreg(1));
    let lhs = Reg::new_vreg(2);
    let rhs = Reg::new_vreg(3);
    let carry = Reg::new_vreg(4);
    let add = function.editor().writer().uaddo(dst, flag, lhs, rhs);
    let adc = function.editor().writer().uadde(dst, flag, lhs, rhs, carry);
    let veloc_lir::InstView::BinaryRegWithFlags(add_view) = function.inst(add).view() else {
        panic!("expected flags");
    };
    let veloc_lir::InstView::BinaryRegWithFlags(adc_view) = function.inst(adc).view() else {
        panic!("expected flags");
    };
    assert_eq!(add_view.carry_in, None);
    assert_eq!(adc_view.carry_in, Some(carry));
    assert_eq!(add_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::Uaddo);
    assert_eq!(adc_view.opcode, veloc_lir::BinaryRegWithFlagsOpcode::Uadde);
    {
        let regs = [lhs, rhs, lhs];
        function.editor().rewriter(add).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Uaddo),
            &[dst.to_reg(), flag.to_reg()],
            &regs,
            &[],
        );
    }
    {
        function.editor().rewriter(adc).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Uadde),
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
    let mut function = MachineFunction::new("test".into());
    let dst = Writable(Reg::new_vreg(0));
    let updated = Writable(Reg::new_vreg(1));
    let base = Reg::new_vreg(2);
    let inst = function
        .editor()
        .writer()
        .indexed_load(dst, updated, base, 16);
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
    let mut function = MachineFunction::new("test".into());
    use veloc_lir::{InstView, SymbolId};
    let results: Vec<_> = (0..8).map(Reg::new_vreg).collect();
    let args: Vec<_> = (8..24).map(Reg::new_vreg).collect();
    let symbol = SymbolId::from_u32(3);
    let direct = function.editor().writer().call(&results, symbol, &args);
    let indirect = function
        .editor()
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
    let ret = function.editor().writer().ret(&args);
    let InstView::Return(view) = function.inst(ret).view() else {
        panic!("expected return");
    };
    assert_eq!(view.values.len(), args.len());
    assert_eq!(view.values.iter().copied().collect::<Vec<_>>(), args);
    let empty = function.editor().writer().ret(&[]);
    let InstView::Return(view) = function.inst(empty).view() else {
        panic!("expected return");
    };
    assert!(view.values.is_empty());
}

#[test]
fn optional_validation_is_separate_from_direct_views() {
    let mut function = MachineFunction::new("test".into());
    use veloc_lir::{MachineOpcode, SymbolId};
    use veloc_mir::{FloatCC, IntCC};
    let dst = Writable(Reg::new_vreg(0));
    let src = Reg::new_vreg(1);
    // Property constraints are defs-driven and remain opt-in.
    let arg = function.editor().writer().arg(dst, -1);
    assert!(matches!(
        function.inst(arg).view(),
        veloc_lir::InstView::Arg(_)
    ));
    assert!(function.inst(arg).validate().is_err());
    function.editor().set_inst_field(arg, 0, InstField::Imm(0));
    function.inst(arg).validate().unwrap();

    let cmp = function.editor().writer().icmp(dst, src, src, IntCC::Eq);
    function
        .editor()
        .set_inst_field(cmp, 0, InstField::FloatCC(FloatCC::Eq));
    assert!(function.inst(cmp).validate().is_err());
    let call = function
        .editor()
        .writer()
        .call(&[dst.to_reg()], SymbolId::from_u32(0), &[src]);
    {
        let mut operands = function.inst(call).fields().to_vec();
        operands.push(InstField::Imm(0));
        function.editor().set_inst_fields(call, &operands);
    }
    assert!(function.inst(call).validate().is_err());
    let missing_callee = function
        .editor()
        .writer()
        .callind(&[dst.to_reg()], src, &[]);
    {
        function.editor().rewriter(missing_callee).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Callind),
            &[dst.to_reg()],
            &[],
            &[],
        );
    }
    assert!(function.inst(missing_callee).validate().is_err());
    let ret = function.editor().writer().ret(&[]);
    {
        function.editor().set_inst_results(ret, &[dst.to_reg()]);
    }
    assert!(function.inst(ret).validate().is_err());
    let target =
        function
            .editor()
            .writer()
            .write(MachineOpcode::Target(0), &[dst.to_reg()], &[src], &[]);
    assert!(function.inst(target).validate().is_err());
    function.editor().invalidate_inst(target);
    assert!(function.inst(target).validate().is_err());

    // Access does not run the optional full shape check: unrelated extra
    // attributes are rejected by validation, not by reading an add's registers.
    let add = function.editor().writer().add(dst, src, src);
    function.editor().set_inst_fields(add, &[InstField::Imm(7)]);
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
