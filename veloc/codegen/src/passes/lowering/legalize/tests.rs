use super::*;
use std::vec::Vec;
use veloc_lir::{GenericOpcode, MachineOpcode};
use veloc_lir::{InstBuild, InstRead};

#[test]
fn x86_displacements_are_checked_and_expansion_preserves_access_metadata() {
    use crate::target::TargetMachine;
    use crate::target::x86_64::X86_64TargetMachine;
    use veloc_lir::{MemoryAccess, MemoryKind};
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
            let block = f.entry_block();
            let base = f.editor().alloc_vreg(Type::PTR);
            let value = f.editor().alloc_vreg(Type::I64);
            let mut memory = MemoryAccess::new(kind, 8);
            memory.alignment = 8;
            memory.volatile = true;
            let inst = match kind {
                MemoryKind::Read => f
                    .editor()
                    .at_end(block)
                    .writer()
                    .with_memory(memory)
                    .load(value, base, offset),
                MemoryKind::Write => f
                    .editor()
                    .at_end(block)
                    .writer()
                    .with_memory(memory)
                    .store(value, base, offset),
            };
            let id = {
                let id = inst;
                id
            };
            Legalizer::new(target.legalizer())
                .legalize(&mut f, |_, _| unreachable!("test contains no calls"))
                .unwrap();
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

#[test]
fn insertion_is_reported() {
    let mut f = MachineFunction::new("insertion".into());
    let block = f.entry_block();
    let (inst, changes) = f.editor().track(|edit| {
        edit.at_end(block)
            .write(MachineOpcode::Generic(GenericOpcode::Add), &[], &[], [])
    });
    assert!(changes.insts.contains(&inst));
}

#[test]
fn existing_value_replacement_updates_users_without_a_copy() {
    use veloc_lir::Type;
    for generated in [false, true] {
        let mut f = MachineFunction::new("replace".into());
        let block = f.entry_block();
        let input = f.editor().alloc_vreg(Type::I64);
        let result = f.editor().alloc_vreg(Type::I64);
        let output = f.editor().alloc_vreg(Type::I64);
        let root = f.editor().at_end(block).writer().copy(result, input);
        let user = f.editor().at_end(block).writer().copy(output, result);

        let ((), changes) = f.editor().track(|edit| {
            let mut ctx = RewriteContext::new(root, edit.editor());
            let input = ctx.inst(ctx.root()).inputs()[0];
            if generated {
                ctx.finish_value(input);
            } else {
                ctx.replace_results(&[input]);
            }
        });
        assert_eq!(f.block_insts(block).collect::<Vec<_>>(), [user]);
        assert_eq!(f.inst(user).inputs(), &[input]);
        assert!(changes.insts.contains(&user));
        assert!(f.inst(root).is_invalid());
        f.check_refs().unwrap();
    }
}
