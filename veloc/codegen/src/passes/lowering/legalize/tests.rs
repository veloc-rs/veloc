use super::*;
use alloc::vec::Vec;
use veloc_lir::{GenericOpcode, MachineOpcode};
use veloc_lir::{InstBuild, InstRead};

#[test]
fn x86_displacements_are_checked_and_expansion_preserves_access_metadata() {
    use crate::target::arch::TargetMachine;
    use crate::target::x86_64::X86_64TargetMachine;
    use veloc_lir::{MemoryAccess, MemoryKind, Writable};
    use veloc_mir::Type;
    let target = X86_64TargetMachine::new(crate::TargetConfig::default()).unwrap();
    for offset in [
        i64::MIN,
        i32::MIN as i64 - 1,
        i32::MIN as i64,
        0,
        i32::MAX as i64,
        i32::MAX as i64 + 1,
        u32::MAX as i64,
        i64::MAX,
    ] {
        for kind in [MemoryKind::Read, MemoryKind::Write] {
            let mut f = MachineFunction::new("offset".into());
            f.editor().create_block();
            let base = f.editor().alloc_vreg(Type::PTR);
            let value = f.editor().alloc_vreg(Type::I64);
            let mut memory = MemoryAccess::new(kind, 8);
            memory.alignment = 8;
            memory.volatile = true;
            let inst = match kind {
                MemoryKind::Read => {
                    f.editor()
                        .writer()
                        .with_memory(memory)
                        .load(Writable(value), base, offset)
                }
                MemoryKind::Write => f
                    .editor()
                    .writer()
                    .with_memory(memory)
                    .store(value, base, offset),
            };
            let id = {
                let id = inst;
                f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
                id
            };
            Legalizer::new(target.legalizer()).legalize(&mut f).unwrap();
            let ids = f
                .block_insts(veloc_lir::BlockId::from_u32(0))
                .collect::<Vec<_>>();
            let expanded = i32::try_from(offset).is_err();
            assert_eq!(ids.len(), if expanded { 3 } else { 1 });
            assert_eq!(ids.last(), Some(&id));
            assert_eq!(f.inst(id).memory(), Some(memory));
            let actual_offset = match f.inst(id).view() {
                veloc_lir::InstView::Load(load) => load.offset,
                veloc_lir::InstView::Store(store) => store.offset,
                _ => panic!("expected offset access"),
            };
            assert_eq!(actual_offset, if expanded { 0 } else { offset });
            if expanded {
                let veloc_lir::InstView::Constant(constant) = f.inst(ids[0]).view() else {
                    panic!("expected constant");
                };
                assert_eq!(constant.imm, offset);
                assert_eq!(f.inst(ids[1]).generic_opcode(), Some(GenericOpcode::PtrAdd));
                assert!(f.inst(ids[0]).memory().is_none() && f.inst(ids[1]).memory().is_none());
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Chain,
    Loop,
    Missing,
    NewBlock,
}

impl TargetLegalizer for Mode {
    fn legalize_action(&self, query: &Query) -> Result<Option<LegalizeAction>> {
        let opcode = query.opcode();
        let apply = match self {
            Self::Loop => |f: &mut RewriteContext<'_>| Mode::Loop.rewrite(f),
            Self::NewBlock => |f: &mut RewriteContext<'_>| Mode::NewBlock.rewrite(f),
            _ => |f: &mut RewriteContext<'_>| Mode::Chain.rewrite(f),
        };
        Ok(match (self, opcode) {
            (Self::Missing, GenericOpcode::Sub) => None,
            (Self::Loop, _) | (_, GenericOpcode::Neg | GenericOpcode::Sub) => {
                Some(LegalizeAction::Rewrite(Rewrite {
                    name: "test",
                    apply,
                }))
            }
            _ => Some(LegalizeAction::Legal),
        })
    }
}
impl Mode {
    fn rewrite(&self, f: &mut RewriteContext<'_>) -> Result<()> {
        let id = f.root();
        if matches!(self, Self::Loop) {
            return Ok(());
        }
        if f.inst(id).generic_opcode() == Some(GenericOpcode::Sub) {
            f.editor().rewriter(id).write(
                MachineOpcode::Generic(GenericOpcode::Add),
                &[],
                &[],
                &[],
            );
            return Ok(());
        }
        let first =
            f.editor()
                .writer()
                .write(MachineOpcode::Generic(GenericOpcode::Sub), &[], &[], &[]);
        if matches!(self, Self::NewBlock) {
            let block = f.editor().create_block();
            f.editor().append_inst(block, first);
            f.replace(&[]);
            return Ok(());
        }
        let second = f.editor().writer().write(
            MachineOpcode::Generic(GenericOpcode::Constant),
            &[],
            &[],
            &[],
        );
        f.replace(&[first, second]);
        Ok(())
    }
}

fn function() -> MachineFunction {
    let mut f = MachineFunction::new("legalize".into());
    f.editor().create_block();
    {
        let id =
            f.editor()
                .writer()
                .write(MachineOpcode::Generic(GenericOpcode::Neg), &[], &[], &[]);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    {
        let id =
            f.editor()
                .writer()
                .write(MachineOpcode::Generic(GenericOpcode::Ret), &[], &[], &[]);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    f
}

#[test]
fn expansions_are_revisited_in_order_including_in_place_changes() {
    for target_node in [false, true] {
        let mut f = function();
        let old = f
            .block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>()[0];
        if target_node {
            f.editor()
                .rewriter(old)
                .write(MachineOpcode::Target(0), &[], &[], &[]);
            assert!(!Legalizer::new(&Mode::Missing).legalize(&mut f).unwrap());
            assert_eq!(f.inst(old).opcode(), MachineOpcode::Target(0));
            continue;
        }
        Legalizer::new(&Mode::Chain).legalize(&mut f).unwrap();
        let ops: alloc::vec::Vec<_> = f
            .block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>()
            .iter()
            .map(|&id| f.inst(id).generic_opcode().unwrap())
            .collect();
        assert_eq!(
            ops,
            [
                GenericOpcode::Add,
                GenericOpcode::Constant,
                GenericOpcode::Ret
            ]
        );
        assert!(f.inst(old).is_invalid());
    }
}

#[test]
fn missing_rules_and_nonconvergent_expansions_are_errors() {
    let error = Legalizer::new(&Mode::Missing)
        .legalize(&mut function())
        .unwrap_err();
    assert!(alloc::format!("{error}").contains("missing legalization rule for Generic(Sub)"));
    let error = Legalizer::new(&Mode::Loop)
        .legalize(&mut function())
        .unwrap_err();
    assert!(alloc::format!("{error}").contains("made no edits"));
}

#[test]
fn blocks_created_by_expansion_are_legalized() {
    let mut f = function();
    Legalizer::new(&Mode::NewBlock).legalize(&mut f).unwrap();
    assert_eq!(f.num_blocks(), 2);
    assert_eq!(
        f.inst(
            f.block_insts(veloc_lir::BlockId::from_u32(1))
                .collect::<Vec<_>>()[0]
        )
        .generic_opcode(),
        Some(GenericOpcode::Add)
    );
}

#[test]
fn edits_to_previously_visited_instructions_are_revisited() {
    struct CrossEdit;
    impl TargetLegalizer for CrossEdit {
        fn legalize_action(&self, query: &Query) -> Result<Option<LegalizeAction>> {
            Ok(Some(match query.opcode() {
                GenericOpcode::Ret => LegalizeAction::rewrite("cross_edit", |ctx| {
                    let first = ctx
                        .blocks()
                        .flat_map(|b| ctx.block_insts(b))
                        .next()
                        .unwrap();
                    ctx.editor().rewriter(first).write(
                        MachineOpcode::Generic(GenericOpcode::Sub),
                        &[],
                        &[],
                        &[],
                    );
                    ctx.replace(&[]);
                    Ok(())
                }),
                GenericOpcode::Sub => LegalizeAction::rewrite("sub_to_add", |ctx| {
                    let root = ctx.root();
                    ctx.editor().rewriter(root).write(
                        MachineOpcode::Generic(GenericOpcode::Add),
                        &[],
                        &[],
                        &[],
                    );
                    Ok(())
                }),
                _ => LegalizeAction::Legal,
            }))
        }
    }
    let mut f = function();
    Legalizer::new(&CrossEdit).legalize(&mut f).unwrap();
    let first = f.blocks().flat_map(|b| f.block_insts(b)).next().unwrap();
    assert_eq!(f.inst(first).generic_opcode(), Some(GenericOpcode::Add));
}

#[test]
fn placement_of_an_existing_detached_instruction_is_reported() {
    let mut f = function();
    let detached =
        f.editor()
            .writer()
            .write(MachineOpcode::Generic(GenericOpcode::Add), &[], &[], &[]);
    let block = f.blocks().next().unwrap();
    let (_, changes) = f
        .editor()
        .track(|f| f.editor().append_inst(block, detached));
    assert!(changes.insts.contains(&detached));
}

#[test]
fn cycles_across_new_blocks_share_one_budget() {
    struct Cycle;
    impl TargetLegalizer for Cycle {
        fn legalize_action(&self, _: &Query) -> Result<Option<LegalizeAction>> {
            Ok(Some(LegalizeAction::rewrite("cycle", |ctx| {
                let root = ctx.root();
                let block = ctx.editor().create_block();
                ctx.editor().detach_inst(root);
                ctx.editor().append_inst(block, root);
                Ok(())
            })))
        }
    }
    let error = Legalizer::new(&Cycle)
        .legalize(&mut function())
        .unwrap_err();
    assert!(alloc::format!("{error}").contains("did not converge"));
}

#[test]
fn existing_value_replacement_updates_users_without_a_copy() {
    use veloc_lir::{Type, Writable};
    for generated in [false, true] {
        let mut f = MachineFunction::new("replace".into());
        let block = f.editor().create_block();
        let input = f.editor().alloc_vreg(Type::I64);
        let result = f.editor().alloc_vreg(Type::I64);
        let output = f.editor().alloc_vreg(Type::I64);
        let root = f.editor().writer().copy(Writable(result), input);
        let user = f.editor().writer().copy(Writable(output), result);
        f.editor().append_inst(block, root);
        f.editor().append_inst(block, user);
        let action = if generated {
            LegalizeAction::rewrite("generated_identity", |ctx| {
                ctx.replace_values(|_, inputs, _, _| inputs[0])
            })
        } else {
            LegalizeAction::rewrite("host_identity", |ctx| {
                let input = ctx.inst(ctx.root()).inputs()[0];
                ctx.replace_results(&[input]);
                Ok(())
            })
        };
        let LegalizeAction::Rewrite(rewrite) = action else {
            unreachable!();
        };
        let (result, changes) = f.editor().track(|edit| rewrite.apply(root, edit));
        result.unwrap();
        assert_eq!(f.block_insts(block).collect::<Vec<_>>(), [user]);
        assert_eq!(f.inst(user).inputs(), &[input]);
        assert!(changes.insts.contains(&user));
        assert!(f.inst(root).is_invalid());
        f.check_refs().unwrap();
    }
}
