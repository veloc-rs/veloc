use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64FrameLowering;

impl TargetFrameLowering for X86_64FrameLowering {
    fn finalize_stack_frame(&self, mfunc: &mut MachineFunction, call_conv: TargetCallConv) {
        let preserved_regs = call_conv.preserved_regs(TargetArch::X86_64);
        let mut used_callee_saved = Vec::new();
        for block in mfunc.blocks() {
            for inst_id in mfunc.block_insts(block) {
                for reg in mfunc.inst(inst_id).defs() {
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

        mfunc.stack_frame.callee_saved_size = (used_callee_saved.len() as u32) * 8;
        mfunc.stack_frame.used_callee_saved = used_callee_saved;

        let mut total = mfunc.stack_frame.local_size
            + mfunc.stack_frame.callee_saved_size
            + mfunc.stack_frame.arg_size;
        let align = 16;
        let misalign = total % align;
        if misalign != 0 {
            total += align - misalign;
        }
        mfunc.stack_frame.total_size = total;
    }

    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction) {
        use crate::target::x86_64::isle::{REG_RBP, REG_RSP, TargetInst};
        use veloc_lir::MachineOpcode;

        let stack_size = mfunc.stack_frame.total_size;
        let saved_regs = mfunc.stack_frame.used_callee_saved.clone();
        let local_size = mfunc.stack_frame.local_size as i32;

        if let Some(entry) = mfunc.entry_block() {
            let mut pending_prologue = Vec::new();

            let push_inst =
                TargetInst::X86PushRbp.write(mfunc.editor().writer(), &[], &[REG_RBP], &[]);
            pending_prologue.push(push_inst);

            let mov_inst = TargetInst::X86MovRbpRsp.write(
                mfunc.editor().writer(),
                &[REG_RBP],
                &[REG_RSP],
                &[],
            );
            pending_prologue.push(mov_inst);

            if stack_size > 0 {
                let sub_inst = TargetInst::X86Sub64ri.write(
                    mfunc.editor().writer(),
                    &[(veloc_lir::Writable(REG_RSP)).to_reg()],
                    &[REG_RSP],
                    &[veloc_lir::InstField::Imm(stack_size as i64)],
                );
                pending_prologue.push(sub_inst);
            }

            for (idx, reg) in saved_regs.iter().copied().enumerate() {
                let offset = -(local_size + ((idx as i32 + 1) * 8));
                let save_inst = TargetInst::X86Store64.write(
                    mfunc.editor().writer(),
                    &[],
                    &[reg, REG_RBP],
                    &[veloc_lir::InstField::Imm(offset as i64)],
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

        let ids: Vec<_> = mfunc.blocks().flat_map(|b| mfunc.block_insts(b)).collect();
        for id in ids {
            let is_ret = matches!(
                mfunc.inst(id).opcode(),
                MachineOpcode::Target(code) if code == TargetInst::X86Ret.as_u32()
            );

            if is_ret {
                for (idx, reg) in saved_regs.iter().copied().enumerate().rev() {
                    let offset = -(local_size + ((idx as i32 + 1) * 8));
                    let restore_inst = TargetInst::X86Load64.write(
                        mfunc.editor().writer(),
                        &[(veloc_lir::Writable(reg)).to_reg()],
                        &[REG_RBP],
                        &[veloc_lir::InstField::Imm(offset as i64)],
                    );
                    mfunc.editor().insert_before(id, restore_inst);
                }

                if stack_size > 0 {
                    let add_inst = TargetInst::X86Add64ri.write(
                        mfunc.editor().writer(),
                        &[(veloc_lir::Writable(REG_RSP)).to_reg()],
                        &[REG_RSP],
                        &[veloc_lir::InstField::Imm(stack_size as i64)],
                    );
                    mfunc.editor().insert_before(id, add_inst);
                }

                let pop_inst =
                    TargetInst::X86PopRbp.write(mfunc.editor().writer(), &[REG_RBP], &[], &[]);
                mfunc.editor().insert_before(id, pop_inst);
            }
        }
    }
}
