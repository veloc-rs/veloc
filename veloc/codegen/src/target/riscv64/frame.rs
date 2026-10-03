use super::*;
use veloc_lir::{FrameLayout, FuncEditor, SavedReg, StackAddress, StackObject};
pub(super) struct Frame;
impl TargetFrameLowering for Frame {
    fn stack_alignment(&self) -> u32 {
        16
    }
    fn finalize_stack_frame(&self, f: &mut FuncEditor<'_>, _: CallConv) -> crate::Result<()> {
        let mut saved = Vec::new();
        let mut outgoing = 0u32;
        for block in f.blocks() {
            for id in f.block_insts(block) {
                if let Some(info) = f.try_call_info(id) {
                    if !saved.contains(&Reg(1)) {
                        saved.push(Reg(1)); // Calls overwrite the link register.
                    }
                    let area = info
                        .frame
                        .and_then(|id| f.stack_frame.call(id))
                        .ok_or_else(|| crate::Error::codegen("call has no ABI frame"))?;
                    if area.align > 16 {
                        return Err(crate::Error::codegen("RV64 call alignment exceeds 16"));
                    }
                    outgoing = outgoing.max(area.size);
                }
                for r in f.inst(id).register_access().writes() {
                    if (r == Reg(1) || ABI.preserved.contains(&r)) && !saved.contains(&r) {
                        saved.push(r);
                    }
                }
            }
        }
        let error =
            || crate::Error::codegen("RV64 frame size or alignment exceeds supported range");
        let frame = &f.stack_frame;
        let mut batch = frame.batch();
        let saves = saved
            .into_iter()
            .map(|reg| SavedReg {
                reg,
                slot: batch.alloc_object(StackObject::Local, 8, 8),
            })
            .collect::<Vec<_>>();
        let mut end = outgoing;
        let mut addresses = cranelift_entity::PrimaryMap::new();
        for slot in frame.slots().values().chain(batch.slots()) {
            if slot.align > 16 {
                return Err(error());
            }
            let offset = match slot.object {
                StackObject::Local => {
                    end = end.checked_add(slot.align - 1).ok_or_else(error)? & !(slot.align - 1);
                    let pos = end;
                    end = end.checked_add(slot.size).ok_or_else(error)?;
                    pos
                }
                StackObject::Incoming { offset } => offset,
                StackObject::Outgoing { offset, .. } => offset,
            };
            addresses.push(StackAddress {
                base: Reg(2),
                offset: i32::try_from(offset).map_err(|_| error())?,
            });
        }
        let total = end.checked_add(15).ok_or_else(error)? & !15;
        i32::try_from(total).map_err(|_| error())?;
        for (slot, data) in frame.slots().iter() {
            if matches!(data.object, StackObject::Incoming { .. }) {
                addresses[slot].offset = addresses[slot]
                    .offset
                    .checked_add(total as i32)
                    .ok_or_else(error)?;
            }
        }
        let callee_saved_size = saves.len() as u32 * 8;
        f.finalize_frame(
            batch,
            FrameLayout {
                addresses,
                saves,
                local_size: end - outgoing - callee_saved_size,
                callee_saved_size,
                total_size: total,
            },
        );
        let mut cursor = veloc_lir::InstCursor::new(f);
        while let Some(id) = cursor.next(f) {
            if f.inst(id).is_call_frame() {
                f.editor().invalidate_inst(id);
            }
        }
        Ok(())
    }
    fn insert_prologue_epilogue(&self, f: &mut FuncEditor<'_>) {
        let layout = f.stack_frame.layout().unwrap();
        let total = layout.total_size as i64;
        if total == 0 {
            return;
        }
        let saves = layout.saves.clone();
        let entry = f.entry_block();
        {
            let mut editor = f.editor();
            let mut at = editor.at_start(entry);
            inst::TargetInst::RvAddOffset.write(
                at.writer(),
                &[Reg(2)],
                &[Reg(2)],
                veloc_lir::Fields::Imm(-total),
            );
            for save in &saves {
                spill_opcode(
                    false,
                    if save.reg.0 >= 32 {
                        Type::F64
                    } else {
                        Type::I64
                    },
                )
                .write(
                    at.writer(),
                    &[],
                    &[save.reg],
                    veloc_lir::Fields::StackMemory {
                        slot: save.slot,
                        flags: veloc_lir::MemFlags::new(),
                    },
                );
            }
        }
        let mut cursor = veloc_lir::InstCursor::new(f);
        while let Some(id) = cursor.next(f) {
            if f.inst(id).opcode() == MachineOpcode::Target(inst::TargetInst::RvRet.as_u32()) {
                let mut editor = f.editor();
                let mut at = editor.before(id);
                for save in saves.iter().rev() {
                    spill_opcode(
                        true,
                        if save.reg.0 >= 32 {
                            Type::F64
                        } else {
                            Type::I64
                        },
                    )
                    .write(
                        at.writer(),
                        &[save.reg],
                        &[],
                        veloc_lir::Fields::StackMemory {
                            slot: save.slot,
                            flags: veloc_lir::MemFlags::new(),
                        },
                    );
                }
                inst::TargetInst::RvAddOffset.write(
                    at.writer(),
                    &[Reg(2)],
                    &[Reg(2)],
                    veloc_lir::Fields::Imm(total),
                );
            }
        }
    }
}
