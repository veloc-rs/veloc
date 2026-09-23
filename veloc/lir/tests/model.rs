//! Construct, decode, validate, and interpret the standalone LIR model.
use veloc_lir::{
    ControlFlow, FieldValue, GenericOpcode, MachineFunction, MachineModule, Reg, RegisterBank,
    SymbolTable, Type, TypeError,
};
use veloc_lir::{InstBuild, InstRead};
use veloc_mir::Linkage;

#[test]
fn function_editor_preserves_layout_and_references() {
    let mut f = MachineFunction::new("layout".into());
    let entry = f.entry_block();
    let exit = f.editor().create_block();
    let a = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(1), &[], &[], []);
    let b = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(2), &[], &[], []);
    f.editor().append_inst(veloc_lir::BlockId::from_u32(0), a);
    f.editor().append_inst(veloc_lir::BlockId::from_u32(0), b);
    f.editor().move_block_before(exit, entry);
    assert_eq!(f.entry_block(), entry);
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
    let (result, _) = f.editor().track(|f| {
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
        .write(veloc_lir::MachineOpcode::Target(3), &[], &[], []);
    let d = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(4), &[], &[], []);
    let (tail, changes) = f.editor().track(|f| {
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
    let edge = f.editor().create_edge(entry, &[]);
    let branch = f.editor().writer().br(edge);
    f.editor().append_inst(exit, branch);
    let (_, changes) = f.editor().track(|f| {
        f.editor().redirect_edge(edge, new);
        f.editor().set_edge_args(edge, &[x]);
    });
    assert!(changes.insts.contains(&branch));
    assert_eq!(f.uses(x).count(), 1);
    let args = f.inst(branch).edge(edge).args.as_ptr();
    f.editor().redirect_edge(edge, entry);
    assert_eq!(f.inst(branch).edge_ids().collect::<Vec<_>>(), [edge]);
    assert_eq!(f.inst(branch).edge(edge).args.as_ptr(), args);
    assert_eq!(f.inst(branch).edge(edge).block, entry);
    f.check_refs().unwrap();
    f.editor().set_edge_args(edge, &[y]);
    assert_eq!(f.uses(x).count(), 0);
    assert_eq!(f.uses(y).count(), 1);
    // Rewriting retains an explicitly reused edge; copying requires a fresh ID.
    f.editor().replace(branch).br(edge);
    assert_eq!(f.inst(branch).edge_ids().collect::<Vec<_>>(), [edge]);
    let copy = f.editor().clone_edge(edge);
    let replacement = f.editor().writer().br(copy);
    assert_ne!(copy, edge);
    assert_eq!(f.uses(y).count(), 2);
    f.check_refs().unwrap();
    // Commit exchanges identities without leaving either instruction dangling.
    f.editor().transfer_edge(edge, copy);
    f.check_refs().unwrap();
    f.editor().replace_inst(branch, replacement);
    assert_eq!(f.inst(branch).edge_ids().collect::<Vec<_>>(), [edge]);
    assert_eq!(f.uses(y).count(), 1);
    f.check_refs().unwrap();
    let cloned = f.clone();
    f.editor().invalidate_inst(branch);
    assert_eq!(f.inst_block(branch), None);
    assert_eq!(f.uses(y).count(), 0);
    assert_eq!(cloned.inst_block(branch), Some(exit));
    assert_eq!(cloned.uses(y).count(), 1);
    f.check_refs().unwrap();
    cloned.check_refs().unwrap();
    // Split builders and reborrowed editors retain the same notification sink.
    let ((created, snapshot), changes) = f.editor().track(|edit| {
        let (mut regs, mut insts) = edit.instruction_parts();
        let dst = regs.alloc(veloc_lir::VRegData {
            ty: Type::I64,
            bank: None,
        });
        let created = insts.writer().copy(dst, x);
        edit.append_inst(entry, created);
        edit.replace_uses(x.as_vreg().unwrap(), y.as_vreg().unwrap());
        (created, MachineFunction::clone(edit))
    });
    assert!(changes.insts.contains(&created));
    assert_eq!(f.inst(created).inputs(), &[y]);
    let mut snapshot = snapshot;
    let (_, changes) = snapshot
        .editor()
        .track(|edit| edit.set_inst_input(created, 0, x));
    assert_eq!(changes.insts, [created]);
    assert_eq!(f.inst(created).inputs(), &[y]);
    f.editor().invalidate_inst(created);

    // RAUW reports every existing owner, not only newly built instructions.
    let user = f.editor().writer().copy(x, y);
    let (_, changes) = f.editor().track(|edit| {
        edit.replace_uses(y.as_vreg().unwrap(), x.as_vreg().unwrap());
    });
    assert!(changes.insts.contains(&user));
    f.editor().invalidate_inst(user);

    // Nested sessions cannot silently divert notifications from the outer one.
    let (_, changes) = f.editor().track(|edit| {
        let nested = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            edit.track(|_| ());
        }));
        assert!(nested.is_err());
        edit.move_before(a, c);
    });
    assert!(changes.insts.contains(&a));

    // Tracking is scoped even when an editor callback panics.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f.editor().track(|_| panic!("abandon edit"));
    }));
    let (_, changes) = f.editor().track(|f| f.editor().move_before(a, c));
    assert!(changes.blocks.contains(&entry));

    // Replacements update the real layout immediately, including retained roots.
    let mut edit = f.editor();
    let replacement = edit
        .writer()
        .write(veloc_lir::MachineOpcode::Target(5), &[], &[], []);
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
        .write(veloc_lir::MachineOpcode::Target(6), &[], &[], []);
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
    use veloc_lir::{RefRole, VReg};
    let mut f = MachineFunction::new("references".into());
    let a = f.editor().alloc_vreg(Type::I64);
    let b = f.editor().alloc_vreg(Type::I64);
    let dst = f.editor().alloc_vreg(Type::I64);
    let block = f.entry_block();
    let add = f.editor().writer().add(dst, a, a);
    let yes = f.editor().create_edge(block, &[a, b]);
    let no = f.editor().create_edge(block, &[a]);
    let branch = f.editor().writer().brcond(a, yes, no);
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
    f.editor().clear_successor_args(branch);
    assert_eq!(f.uses(b).count(), 3);
    assert_eq!(clone.uses(b).count(), 6);
    clone.check_refs().unwrap();

    // Replacing a tied input never renames its independent output.
    let rw = f
        .editor()
        .writer()
        .write(veloc_lir::MachineOpcode::Target(0), &[a], &[b, a], []);
    f.editor()
        .replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
    assert_eq!(f.uses(a).count(), 0);
    assert_eq!(f.inst(rw).defs().collect::<Vec<_>>(), [a]);
    f.editor().invalidate_inst(rw);
    f.check_refs().unwrap();

    let replacement = f.editor().writer().copy(dst, a);
    f.editor().replace_inst(add, replacement);
    assert!(f.inst(replacement).is_invalid());
    assert_eq!(f.uses(a).single().unwrap().inst(), add);
    assert_eq!(f.defs(dst).single().unwrap().inst(), add);
    f.check_refs().unwrap();

    // Both role changes and pooled-range reuse must unlink obsolete entries.
    for index in 0..128 {
        f.editor().replace(add).constant(dst, index);
        assert_eq!(f.uses(a).count(), 0);
        f.editor().replace(add).add(dst, a, a);
        assert_eq!(f.uses(a).count(), 2);
        f.check_refs().unwrap();
    }
    // Register references and attribute edits address independent storage domains.
    let mixed = f.editor().writer().write(
        veloc_lir::MachineOpcode::Target(0),
        &[],
        &[a, a],
        [FieldValue::Imm(7)],
    );
    assert_eq!(f.inst(mixed).inputs(), &[a, a]);
    assert_eq!(f.inst(mixed).fields().len(), 1);
    f.editor().set_inst_input(mixed, 1, b);
    assert_eq!(f.inst(mixed).inputs()[1], b);
    f.check_refs().unwrap();
    f.editor()
        .replace_uses(a.as_vreg().unwrap(), b.as_vreg().unwrap());
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
    let edges = [
        f.editor().create_edge(block, &[a, a, b]),
        f.editor().create_edge(block, &[]),
        f.editor().create_edge(block, &[a]),
    ];
    let fields = edges.map(veloc_lir::FieldValue::Edge);
    let source =
        f.editor()
            .writer()
            .write(veloc_lir::MachineOpcode::Target(42), &[dst], &[a], fields);
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
    assert_eq!(
        f.successors(source)
            .map(|t| t.args.to_vec())
            .collect::<Vec<_>>(),
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
    let block = function.entry_block();
    let reg = function.editor().alloc_vreg(Type::I64);
    let inst = function.editor().writer().constant(reg, 42);
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
    let mut function = MachineFunction::new("edit".into());
    let block = function.entry_block();
    let edge = function.editor().create_edge(block, &[]);
    let id = function.editor().writer().br(edge);
    function.editor().redirect_edge(edge, block);
    assert_eq!(function.successors(id).next().unwrap().block, block);

    function.editor().invalidate_inst(id);
    assert!(function.successors(id).next().is_none());
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
        .constant(veloc_lir::Reg::new_vreg(0), 42);
    {
        function.editor().replace(inst).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Constant),
            &[],
            &[],
            [veloc_lir::FieldValue::Imm(42)],
        );
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
    let inst = function
        .editor()
        .writer()
        .add(Reg::new_vreg(0), Reg::new_vreg(1), Reg::new_vreg(2));
    assert!(matches!(
        function.inst(inst).view(),
        veloc_lir::InstView::BinaryReg(_)
    ));
}

#[test]
fn generated_builders_and_views_agree() {
    let mut function = MachineFunction::new("test".into());
    let dst = Reg::new_vreg(0);
    let lhs = Reg::new_vreg(1);
    let rhs = Reg::new_vreg(2);
    let veloc_lir::InstView::BinaryReg(decoded) = ({
        let id = function.editor().writer().add(dst, lhs, rhs);
        function.inst(id).view()
    }) else {
        panic!("expected BinaryReg");
    };
    assert_eq!((decoded.dst, decoded.lhs, decoded.rhs), (dst, lhs, rhs));
    assert_eq!(decoded.opcode, veloc_lir::BinaryRegOpcode::Add);
    assert_eq!(GenericOpcode::Add.control(), ControlFlow::Next);
    assert_eq!(GenericOpcode::Brcond.control(), ControlFlow::Jump);
    assert_eq!(GenericOpcode::Trap.control(), ControlFlow::Trap);
}

#[test]
fn carry_input_is_required_exactly_for_carry_instructions() {
    let mut function = MachineFunction::new("test".into());
    let dst = Reg::new_vreg(0);
    let flag = Reg::new_vreg(1);
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
        function.editor().replace(add).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Uaddo),
            &[dst, flag],
            &regs,
            [],
        );
    }
    {
        function.editor().replace(adc).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Uadde),
            &[dst, flag],
            &[lhs, rhs],
            [],
        );
    }
    assert!(function.inst(add).validate().is_err());
    assert!(function.inst(adc).validate().is_err());
}

#[test]
fn variable_views_preserve_call_and_return_operands() {
    let mut function = MachineFunction::new("test".into());
    use veloc_lir::{InstView, SymbolId};
    let results: Vec<_> = (0..8).map(Reg::new_vreg).collect();
    let args: Vec<_> = (8..24).map(Reg::new_vreg).collect();
    let symbol = SymbolId::from_u32(3);
    let info = veloc_lir::CallInfo {
        clobbers: veloc_lir::RegMask::from_static(&[1 << 3, 1]),
        sig: veloc_mir::Signature::new(
            vec![Type::I64; args.len()],
            vec![Type::I64; results.len()],
            veloc_mir::CallConv::SystemV,
        ),
        frame: None,
        stack_args: Default::default(),
    };
    let direct = function
        .editor()
        .writer()
        .call(&results, symbol, &args, info.clone());
    let indirect =
        function
            .editor()
            .writer()
            .callind(&results, Reg::new_vreg(25), &args, info.clone());
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
    assert_eq!(
        function.inst(direct).clobbers().collect::<Vec<_>>(),
        [Reg::new_preg(3), Reg::new_preg(64)]
    );
    assert_eq!(function.defs(Reg::new_preg(3)).count(), 0);
    assert!(info.clobbers.contains(veloc_lir::PReg::new(64)));
    assert!(!info.clobbers.contains(veloc_lir::PReg::new(63)));
    // Each instruction owns its call contract. Views borrow it directly.
    let direct_info = match function.inst(direct).view() {
        InstView::Call(call) => call.info,
        _ => unreachable!(),
    };
    let indirect_info = match function.inst(indirect).view() {
        InstView::CallIndirect(call) => call.info,
        _ => unreachable!(),
    };
    assert!(!core::ptr::eq(direct_info, indirect_info));
    let indirect_info = indirect_info.clone();
    let mut lowered = info.clone();
    lowered.frame = Some(function.editor().alloc_call_frame(veloc_lir::StackArea {
        size: 32,
        align: 16,
    }));
    function
        .editor()
        .replace(direct)
        .call(&results, symbol, &args, lowered);
    function.check_refs().unwrap();
    assert_eq!(
        function
            .stack_frame
            .call(function.call_info(direct).frame.unwrap())
            .unwrap()
            .size,
        32
    );
    assert!(function.call_info(indirect).frame.is_none());
    // Replacing the call with a non-call drops only its own contract.
    function.editor().replace(direct).ret(&[]);
    assert!(function.try_call_info(direct).is_none());
    assert!(function.call_info(indirect).frame.is_none());

    function.editor().replace_inst(direct, indirect);
    let InstView::CallIndirect(moved) = function.inst(direct).view() else {
        panic!("expected moved call")
    };
    assert_eq!(moved.info, &indirect_info);
    assert!(function.try_call_info(indirect).is_none());
    let mut lowered = indirect_info.clone();
    lowered.frame = Some(function.editor().alloc_call_frame(veloc_lir::StackArea {
        size: 16,
        align: 16,
    }));
    function
        .editor()
        .replace(direct)
        .callind(&results, Reg::new_vreg(25), &args, lowered);
    function.check_refs().unwrap();
    assert_eq!(
        function
            .stack_frame
            .call(function.call_info(direct).frame.unwrap())
            .unwrap()
            .size,
        16
    );

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
    let dst = Reg::new_vreg(0);
    let src = Reg::new_vreg(1);
    let cmp = function.editor().writer().icmp(dst, src, src, IntCC::Eq);
    // Structural validation remains explicit, including low-level writes.
    function.editor().replace(cmp).write(
        MachineOpcode::Generic(GenericOpcode::Icmp),
        &[dst],
        &[src, src],
        [FieldValue::FloatCC(FloatCC::Eq)],
    );
    assert!(function.inst(cmp).validate().is_err());
    let info = veloc_lir::CallInfo {
        clobbers: Default::default(),
        sig: veloc_mir::Signature::new([Type::I64], [Type::I64], veloc_mir::CallConv::SystemV),
        frame: None,
        stack_args: Default::default(),
    };
    let call = function
        .editor()
        .writer()
        .call(&[dst], SymbolId::from_u32(0), &[src], info.clone());
    function.inst(call).validate().unwrap();
    // A second, unrelated payload is not representable. Reject it before
    // changing the instruction or its owned call contract.
    let mut fields = vec![
        FieldValue::Global(SymbolId::from_u32(0)),
        FieldValue::Call(info.clone()),
    ];
    fields.push(FieldValue::Imm(0));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            function.editor().replace(call).write(
                MachineOpcode::Generic(GenericOpcode::Call),
                &[dst],
                &[src],
                fields,
            );
        }))
        .is_err()
    );
    function.inst(call).validate().unwrap();
    assert_eq!(function.call_info(call), &info);
    let missing_callee = function
        .editor()
        .writer()
        .callind(&[dst], src, &[], info.clone());
    {
        function.editor().replace(missing_callee).write(
            veloc_lir::MachineOpcode::Generic(GenericOpcode::Callind),
            &[dst],
            &[],
            [FieldValue::Call(info)],
        );
    }
    assert!(function.inst(missing_callee).validate().is_err());
    let ret = function.editor().writer().ret(&[]);
    {
        function.editor().set_inst_results(ret, &[dst]);
    }
    assert!(function.inst(ret).validate().is_err());
    let target = function
        .editor()
        .writer()
        .write(MachineOpcode::Target(0), &[dst], &[src], []);
    assert!(function.inst(target).validate().is_err());
    function.editor().invalidate_inst(target);
    assert!(function.inst(target).validate().is_err());

    // Access does not run the optional full shape check: unrelated extra
    // attributes are rejected by validation, not by reading an add's registers.
    let add = function.editor().writer().add(dst, src, src);
    function.editor().replace(add).write(
        MachineOpcode::Generic(GenericOpcode::Add),
        &[dst],
        &[src, src],
        [veloc_lir::FieldValue::Imm(7)],
    );
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
