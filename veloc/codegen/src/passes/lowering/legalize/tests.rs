use super::*;
use alloc::vec;
use veloc_lir::{InstId, MachineBlock, MachineOpcode};
use veloc_mir::Block;

#[test]
fn x86_displacements_are_checked_and_expansion_preserves_access_metadata() {
    use crate::target::arch::TargetMachine;
    use crate::target::x86_64::X86_64TargetMachine;
    use veloc_lir::{MemoryAccess, MemoryKind, Writable};
    use veloc_mir::Type;
    let target = X86_64TargetMachine::new(crate::TargetConfig::default());
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
            let mut f = MachineFunction::<LegalizedLir>::new("offset".into());
            f.blocks.push(MachineBlock::new(Block(0)));
            let base = f.alloc_vreg(Type::PTR);
            let value = f.alloc_vreg(Type::I64);
            let mut memory = MemoryAccess::new(kind, 8);
            memory.alignment = 8;
            memory.volatile = true;
            let inst = match kind {
                MemoryKind::Read => MachineInst::build_offset_load(Writable(value), base, offset),
                MemoryKind::Write => MachineInst::build_offset_store(value, base, offset),
            }
            .with_memory(memory);
            let id = f.alloc_inst_and_append_to_block(0, inst);
            Legalizer::new(target.target_legalizer())
                .legalize(&mut f)
                .unwrap();
            let ids = f.block_insts(0);
            let expanded = i32::try_from(offset).is_err();
            assert_eq!(ids.len(), if expanded { 3 } else { 1 });
            assert_eq!(ids.last(), Some(&id));
            assert_eq!(f.dfg[id].memory, Some(memory));
            let actual_offset = match f.dfg[id].generic_view().unwrap() {
                veloc_lir::InstView::LoadOffset(load) => load.offset,
                veloc_lir::InstView::StoreOffset(store) => store.offset,
                _ => panic!("expected offset access"),
            };
            assert_eq!(actual_offset, if expanded { 0 } else { offset });
            if expanded {
                let veloc_lir::InstView::Constant(constant) = f.dfg[ids[0]].generic_view().unwrap()
                else {
                    panic!("expected constant");
                };
                assert_eq!(constant.imm, offset);
                assert_eq!(
                    f.dfg[ids[1]].generic_opcode(),
                    Some(GenericOpcode::G_PTR_ADD)
                );
                assert!(f.dfg[ids[0]].memory.is_none() && f.dfg[ids[1]].memory.is_none());
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

fn inst(op: GenericOpcode) -> MachineInst {
    MachineInst::build_generic(MachineOpcode::Generic(op), smallvec::SmallVec::new())
}

impl TargetLegalizer for Mode {
    fn legalize_action(
        &self,
        i: &MachineInst,
        _: &MachineFunction<LegalizedLir>,
    ) -> Result<Option<LegalizeAction>> {
        match (self, i.generic_opcode().unwrap()) {
            (Self::Missing, GenericOpcode::G_SUB) => Ok(None),
            (Self::Loop, _) | (_, GenericOpcode::G_NEG | GenericOpcode::G_SUB) => {
                Ok(Some(LegalizeAction::Lower))
            }
            _ => Ok(Some(LegalizeAction::Legal)),
        }
    }

    fn legalize_instruction(
        &self,
        id: InstId,
        f: &mut MachineFunction<LegalizedLir>,
    ) -> Result<LegalizeResult> {
        if matches!(self, Self::Loop) {
            return Ok(LegalizeResult::Replace(vec![id]));
        }
        if f.dfg[id].generic_opcode() == Some(GenericOpcode::G_SUB) {
            f.replace_inst(id, inst(GenericOpcode::G_ADD));
            return Ok(LegalizeResult::Replace(vec![id]));
        }
        let first = f.alloc_inst(inst(GenericOpcode::G_SUB));
        if matches!(self, Self::NewBlock) {
            f.create_synthetic_block();
            f.append_inst_id_to_block(f.num_blocks() - 1, first);
            return Ok(LegalizeResult::Replace(vec![]));
        }
        let second = f.alloc_inst(inst(GenericOpcode::G_CONSTANT));
        Ok(LegalizeResult::Replace(vec![first, second]))
    }
}

fn function() -> MachineFunction<LegalizedLir> {
    let mut f = MachineFunction::new("legalize".into());
    f.blocks.push(MachineBlock::new(Block(0)));
    f.alloc_inst_and_append_to_block(0, inst(GenericOpcode::G_NEG));
    f.alloc_inst_and_append_to_block(0, inst(GenericOpcode::G_RET));
    f
}

#[test]
fn expansions_are_revisited_in_order_including_in_place_changes() {
    let mut f = function();
    let old = f.block_insts(0)[0];
    Legalizer::new(&Mode::Chain).legalize(&mut f).unwrap();
    let ops: alloc::vec::Vec<_> = f
        .block_insts(0)
        .iter()
        .map(|&id| f.dfg[id].generic_opcode().unwrap())
        .collect();
    assert_eq!(
        ops,
        [
            GenericOpcode::G_ADD,
            GenericOpcode::G_CONSTANT,
            GenericOpcode::G_RET
        ]
    );
    assert!(f.dfg[old].is_invalid());
}

#[test]
fn missing_rules_and_nonconvergent_expansions_are_errors() {
    let error = Legalizer::new(&Mode::Missing)
        .legalize(&mut function())
        .unwrap_err();
    assert!(alloc::format!("{error}").contains("missing legalization rule for G_SUB"));
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
        f.dfg[f.block_insts(1)[0]].generic_opcode(),
        Some(GenericOpcode::G_ADD)
    );
}
