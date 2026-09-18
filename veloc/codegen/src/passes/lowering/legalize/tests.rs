use super::*;
use alloc::vec;
use alloc::vec::Vec;
use veloc_lir::{GenericOpcode, InstId, MachineOpcode};
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
                MemoryKind::Read => f.editor().writer().with_memory(memory).offset_load(
                    Writable(value),
                    base,
                    offset,
                ),
                MemoryKind::Write => f
                    .editor()
                    .writer()
                    .with_memory(memory)
                    .offset_store(value, base, offset),
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
                veloc_lir::InstView::LoadOffset(load) => load.offset,
                veloc_lir::InstView::StoreOffset(store) => store.offset,
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
    fn legalize_target(&self, _: &veloc_lir::InstRef<'_>) -> Result<Option<LegalizeAction>> {
        Ok(match self {
            Self::Missing => None,
            _ => Some(LegalizeAction::Rewrite(Rewrite {
                name: "target_expansion",
                apply: |id, f| Mode::Chain.rewrite(id, f),
            })),
        })
    }
    fn legalize_action(&self, query: &Query) -> Result<Option<LegalizeAction>> {
        let opcode = query.opcode;
        let apply = match self {
            Self::Loop => |id, f: &mut MachineFunction| Mode::Loop.rewrite(id, f),
            Self::NewBlock => |id, f: &mut MachineFunction| Mode::NewBlock.rewrite(id, f),
            _ => |id, f: &mut MachineFunction| Mode::Chain.rewrite(id, f),
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
    fn rewrite(&self, id: InstId, f: &mut MachineFunction) -> Result<LegalizeResult> {
        if matches!(self, Self::Loop) {
            return Ok(LegalizeResult::Replace(vec![id]));
        }
        if f.inst(id).generic_opcode() == Some(GenericOpcode::Sub) {
            f.editor().rewriter(id).write(
                MachineOpcode::Generic(GenericOpcode::Add),
                &[],
                &[],
                &[],
            );
            return Ok(LegalizeResult::Replace(vec![id]));
        }
        let first =
            f.editor()
                .writer()
                .write(MachineOpcode::Generic(GenericOpcode::Sub), &[], &[], &[]);
        if matches!(self, Self::NewBlock) {
            let block = f.editor().create_block();
            f.editor().append_inst(block, first);
            return Ok(LegalizeResult::Replace(vec![]));
        }
        let second = f.editor().writer().write(
            MachineOpcode::Generic(GenericOpcode::Constant),
            &[],
            &[],
            &[],
        );
        Ok(LegalizeResult::Replace(vec![first, second]))
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
            let error = Legalizer::new(&Mode::Missing).legalize(&mut f).unwrap_err();
            assert!(alloc::format!("{error}").contains("missing legalization rule for Target(0)"));
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
    assert!(alloc::format!("{error}").contains("did not converge"));
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
