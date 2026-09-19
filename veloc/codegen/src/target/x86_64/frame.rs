use super::inst as generated;
use crate::target::{CallConv as TargetCallConv, TargetArch, TargetFrameLowering};
use alloc::vec::Vec;
use veloc_lir::MachineFunction;

#[derive(Debug, Clone, Copy)]
pub struct X86_64FrameLowering;

impl TargetFrameLowering for X86_64FrameLowering {
    fn stack_alignment(&self) -> u32 {
        16
    }

    fn finalize_stack_frame(
        &self,
        mfunc: &mut MachineFunction,
        call_conv: TargetCallConv,
    ) -> crate::Result<()> {
        let preserved_regs = call_conv.preserved_regs(TargetArch::X86_64);
        let mut used_callee_saved = Vec::new();
        let align = self.stack_alignment();
        let mut outgoing = 0;

        for block in mfunc.blocks() {
            for inst_id in mfunc.block_insts(block) {
                if let Some(info) = mfunc.try_call_info(inst_id) {
                    let stack = info
                        .stack
                        .ok_or_else(|| crate::Error::codegen("call has not been ABI lowered"))?;
                    if stack.align > align {
                        return Err(crate::Error::codegen("unsupported call stack alignment"));
                    }
                    outgoing = outgoing.max(stack.size);
                }
                for reg in mfunc
                    .inst(inst_id)
                    .defs()
                    .chain(mfunc.inst(inst_id).clobbers())
                {
                    if reg != generated::REG_RBP
                        && reg != generated::REG_RSP
                        && preserved_regs.contains(&reg)
                        && !used_callee_saved.contains(&reg)
                    {
                        used_callee_saved.push(reg);
                    }
                }
            }
        }

        use veloc_lir::{FrameLayout, SavedReg, StackAddress, StackObject};
        let error = || crate::Error::codegen("unsupported frame size or alignment");
        // Build off to the side: failure must not install a partial layout.
        let frame = &mfunc.stack_frame;
        let mut batch = frame.batch();
        let original_slots = frame.slots().len();
        let mut saves = Vec::new();
        for reg in used_callee_saved {
            // This frame implementation currently saves GPRs only. Do not
            // silently truncate a preserved SIMD register to eight bytes.
            if reg.0 >= 16 {
                return Err(crate::Error::codegen(
                    "SIMD callee-save lowering is not implemented",
                ));
            }
            let slot = batch.alloc_object(StackObject::Local, 8, 8);
            saves.push(SavedReg { reg, slot });
        }
        let mut addresses = cranelift_entity::PrimaryMap::new();
        let mut end: u32 = 0;
        let mut local_size = 0;
        for (index, slot) in frame.slots().values().chain(batch.slots()).enumerate() {
            if index == original_slots {
                local_size = end;
            }
            if slot.align > align {
                return Err(error());
            }
            let address = match slot.object {
                StackObject::Local => {
                    end = end
                        .checked_add(slot.size)
                        .and_then(|n| n.checked_add(slot.align - 1))
                        .ok_or_else(error)?
                        & !(slot.align - 1);
                    StackAddress {
                        base: generated::REG_RBP,
                        offset: -i32::try_from(end).map_err(|_| error())?,
                    }
                }
                StackObject::Incoming { offset } => {
                    // Return address and pushed RBP precede the argument area.
                    let offset = offset.checked_add(16).ok_or_else(error)?;
                    StackAddress {
                        base: generated::REG_RBP,
                        offset: i32::try_from(offset).map_err(|_| error())?,
                    }
                }
                StackObject::Outgoing { offset } => {
                    outgoing = outgoing.max(offset.checked_add(slot.size).ok_or_else(error)?);
                    StackAddress {
                        base: generated::REG_RSP,
                        offset: i32::try_from(offset).map_err(|_| error())?,
                    }
                }
            };
            addresses.push(address);
        }
        if saves.is_empty() {
            local_size = end;
        }
        let total_size = end
            .checked_add(outgoing)
            .and_then(|n| n.checked_add(align - 1))
            .ok_or_else(error)?
            & !(align - 1);
        i32::try_from(total_size).map_err(|_| error())?;
        let layout = FrameLayout {
            addresses,
            saves,
            local_size,
            callee_saved_size: end - local_size,
            total_size,
        };
        mfunc.stack_frame.append(batch);
        mfunc.stack_frame.finish(layout);
        Ok(())
    }

    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction) {
        use crate::target::x86_64::inst::{REG_RBP, REG_RSP, TargetInst};
        use veloc_lir::MachineOpcode;

        let layout = mfunc.stack_frame.layout().expect("finalized frame");
        let stack_size = layout.total_size;
        let saved_count = layout.saves.len();

        {
            let entry = mfunc.entry_block();
            let mut pending_prologue = Vec::new();

            let push_inst =
                TargetInst::X86PushRbp.write(mfunc.editor().writer(), &[], &[REG_RBP], []);
            pending_prologue.push(push_inst);

            let mov_inst =
                TargetInst::X86MovRbpRsp.write(mfunc.editor().writer(), &[REG_RBP], &[REG_RSP], []);
            pending_prologue.push(mov_inst);

            if stack_size > 0 {
                let sub_inst = TargetInst::X86Sub64ri.write(
                    mfunc.editor().writer(),
                    &[(veloc_lir::Writable(REG_RSP)).to_reg()],
                    &[REG_RSP],
                    [veloc_lir::FieldValue::Imm(stack_size as i64)],
                );
                pending_prologue.push(sub_inst);
            }

            for index in 0..saved_count {
                let save = mfunc.stack_frame.layout().unwrap().saves[index];
                let save_inst = TargetInst::X86Store64Stack.write(
                    mfunc.editor().writer(),
                    &[],
                    &[save.reg],
                    [veloc_lir::FieldValue::StackSlot(save.slot)],
                );
                pending_prologue.push(save_inst);
            }

            let first = mfunc.layout().first_inst(entry);
            let mut edit = mfunc.editor();
            for inst in pending_prologue {
                if let Some(first) = first {
                    edit.insert_before(first, inst);
                } else {
                    edit.append_inst(entry, inst);
                }
            }
        }

        let mut cursor = veloc_lir::InstCursor::new(mfunc);
        while let Some(id) = cursor.next(mfunc) {
            let is_ret = matches!(
                mfunc.inst(id).opcode(),
                MachineOpcode::Target(code) if code == TargetInst::X86Ret.as_u32()
            );

            if is_ret {
                for index in (0..saved_count).rev() {
                    let save = mfunc.stack_frame.layout().unwrap().saves[index];
                    let restore_inst = TargetInst::X86Load64Stack.write(
                        mfunc.editor().writer(),
                        &[save.reg],
                        &[],
                        [veloc_lir::FieldValue::StackSlot(save.slot)],
                    );
                    mfunc.editor().insert_before(id, restore_inst);
                }

                if stack_size > 0 {
                    let add_inst = TargetInst::X86Add64ri.write(
                        mfunc.editor().writer(),
                        &[(veloc_lir::Writable(REG_RSP)).to_reg()],
                        &[REG_RSP],
                        [veloc_lir::FieldValue::Imm(stack_size as i64)],
                    );
                    mfunc.editor().insert_before(id, add_inst);
                }

                let pop_inst =
                    TargetInst::X86PopRbp.write(mfunc.editor().writer(), &[REG_RBP], &[], []);
                mfunc.editor().insert_before(id, pop_inst);
            }
        }
    }
}
