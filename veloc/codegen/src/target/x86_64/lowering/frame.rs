use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64FrameLowering;

impl TargetFrameLowering for X86_64FrameLowering {
    fn finalize_stack_frame(&self, mfunc: &mut MachineFunction, call_conv: TargetCallConv) {
        let preserved_regs = call_conv.preserved_regs(TargetArch::X86_64);
        let mut used_callee_saved = Vec::new();
        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
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

        if !mfunc.blocks.is_empty() {
            let mut pending_prologue = Vec::new();

            let push_inst = mfunc.writer().write(
                MachineOpcode::Target(TargetInst::X86PushRbp.as_u32()),
                &[],
                &[REG_RBP],
                &[],
            );
            pending_prologue.push(push_inst);

            let mov_inst = mfunc.writer().write(
                MachineOpcode::Target(TargetInst::X86MovRbpRsp.as_u32()),
                &[REG_RBP],
                &[REG_RSP],
                &[],
            );
            pending_prologue.push(mov_inst);

            if stack_size > 0 {
                let sub_inst = mfunc.writer().write(
                    MachineOpcode::Target(TargetInst::X86Sub64ri.as_u32()),
                    &[(veloc_lir::Writable(REG_RSP)).to_reg()],
                    &[REG_RSP],
                    &[veloc_lir::InstField::Imm(stack_size as i64)],
                );
                pending_prologue.push(sub_inst);
            }

            for (idx, reg) in saved_regs.iter().copied().enumerate() {
                let offset = -(local_size + ((idx as i32 + 1) * 8));
                let save_inst = mfunc.writer().write(
                    MachineOpcode::Target(TargetInst::X86Store64.as_u32()),
                    &[],
                    &[reg, REG_RBP],
                    &[veloc_lir::InstField::Imm(offset as i64)],
                );
                pending_prologue.push(save_inst);
            }

            mfunc
                .rewrite_block(0, |cursor| {
                    if !pending_prologue.is_empty() {
                        for inst_id in pending_prologue.drain(..) {
                            cursor.emit(inst_id);
                        }
                    }
                    cursor.keep_current();
                    Ok::<(), crate::error::Error>(())
                })
                .expect("x86_64 prologue rewriting should not fail");
        }

        for block_idx in 0..mfunc.blocks.len() {
            mfunc
                .rewrite_block(block_idx, |cursor| {
                    let is_ret = matches!(
                        cursor.current_inst().opcode(),
                        MachineOpcode::Target(code) if code == TargetInst::X86Ret.as_u32()
                    );

                    if is_ret {
                        for (idx, reg) in saved_regs.iter().copied().enumerate().rev() {
                            let offset = -(local_size + ((idx as i32 + 1) * 8));
                            let restore_inst = cursor.mfunc_mut().writer().write(
                                MachineOpcode::Target(TargetInst::X86Load64.as_u32()),
                                &[(veloc_lir::Writable(reg)).to_reg()],
                                &[REG_RBP],
                                &[veloc_lir::InstField::Imm(offset as i64)],
                            );
                            cursor.emit(restore_inst);
                        }

                        if stack_size > 0 {
                            let add_inst = cursor.mfunc_mut().writer().write(
                                MachineOpcode::Target(TargetInst::X86Add64ri.as_u32()),
                                &[(veloc_lir::Writable(REG_RSP)).to_reg()],
                                &[REG_RSP],
                                &[veloc_lir::InstField::Imm(stack_size as i64)],
                            );
                            cursor.emit(add_inst);
                        }

                        let pop_inst = cursor.mfunc_mut().writer().write(
                            MachineOpcode::Target(TargetInst::X86PopRbp.as_u32()),
                            &[REG_RBP],
                            &[],
                            &[],
                        );
                        cursor.emit(pop_inst);
                    }

                    cursor.keep_current();
                    Ok::<(), crate::error::Error>(())
                })
                .expect("x86_64 prologue/epilogue rewriting should not fail");
        }
    }
}
