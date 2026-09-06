use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64FrameLowering;

impl X86_64FrameLowering {
    pub fn new(_cpu: CpuDescription) -> Self {
        Self
    }
}

impl TargetFrameLowering for X86_64FrameLowering {
    fn finalize_stack_frame(
        &self,
        mfunc: &mut MachineFunction<RegAllocated>,
        call_conv: TargetCallConv,
    ) {
        let preserved_regs = call_conv.preserved_regs(TargetArch::X86_64);
        let mut used_callee_saved = Vec::new();
        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
                for reg in mfunc.dfg[inst_id].defs().chain(mfunc.dfg[inst_id].uses()) {
                    if preserved_regs.contains(&reg) && !used_callee_saved.contains(&reg) {
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

    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction<RegAllocated>) {
        use crate::lir::MachineOpcode;
        use crate::target::x86_64::isle::{REG_RBP, REG_RSP, TargetInst};

        let stack_size = mfunc.stack_frame.total_size;
        let saved_regs = mfunc.stack_frame.used_callee_saved.clone();
        let local_size = mfunc.stack_frame.local_size as i32;

        if !mfunc.blocks.is_empty() {
            let mut pending_prologue = Vec::new();

            let push_inst = MachineInst::build_generic(
                MachineOpcode::Target(TargetInst::X86PushRbp.as_u32()),
                smallvec::SmallVec::new(),
            );
            pending_prologue.push(mfunc.alloc_inst(push_inst));

            let mov_inst = MachineInst::build_generic(
                MachineOpcode::Target(TargetInst::X86MovRbpRsp.as_u32()),
                smallvec::SmallVec::new(),
            );
            pending_prologue.push(mfunc.alloc_inst(mov_inst));

            if stack_size > 0 {
                let sub_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Sub64ri.as_u32()),
                    smallvec::smallvec![
                        crate::lir::MachineOperand::TiedDefUse(crate::lir::Writable(REG_RSP)),
                        crate::lir::MachineOperand::Imm(stack_size as i64),
                    ],
                );
                pending_prologue.push(mfunc.alloc_inst(sub_inst));
            }

            for (idx, reg) in saved_regs.iter().copied().enumerate() {
                let offset = -(local_size + ((idx as i32 + 1) * 8));
                let save_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Store64.as_u32()),
                    smallvec::smallvec![
                        crate::lir::MachineOperand::Use(reg),
                        crate::lir::MachineOperand::Use(REG_RBP),
                        crate::lir::MachineOperand::Imm(offset as i64),
                    ],
                );
                pending_prologue.push(mfunc.alloc_inst(save_inst));
            }

            mfunc
                .rewrite_block(0, |cursor| {
                    if !pending_prologue.is_empty() {
                        for inst_id in pending_prologue.drain(..) {
                            cursor.emit_existing_before(inst_id);
                        }
                    }
                    cursor.emit_existing_before(cursor.current_inst_id());
                    cursor.remove_current();
                    Ok::<(), crate::error::Error>(())
                })
                .expect("x86_64 prologue rewriting should not fail");
        }

        for block_idx in 0..mfunc.blocks.len() {
            mfunc
                .rewrite_block(block_idx, |cursor| {
                    let inst_id = cursor.current_inst_id();
                    let is_ret = matches!(
                        cursor.current_inst().opcode,
                        MachineOpcode::Target(code) if code == TargetInst::X86Ret.as_u32()
                    );

                    if is_ret {
                        for (idx, reg) in saved_regs.iter().copied().enumerate().rev() {
                            let offset = -(local_size + ((idx as i32 + 1) * 8));
                            let restore_inst = MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86Load64.as_u32()),
                                smallvec::smallvec![
                                    crate::lir::MachineOperand::Def(crate::lir::Writable(reg)),
                                    crate::lir::MachineOperand::Use(REG_RBP),
                                    crate::lir::MachineOperand::Imm(offset as i64),
                                ],
                            );
                            cursor.emit_before(restore_inst);
                        }

                        if stack_size > 0 {
                            let add_inst = MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86Add64ri.as_u32()),
                                smallvec::smallvec![
                                    crate::lir::MachineOperand::TiedDefUse(crate::lir::Writable(
                                        REG_RSP
                                    )),
                                    crate::lir::MachineOperand::Imm(stack_size as i64),
                                ],
                            );
                            cursor.emit_before(add_inst);
                        }

                        let pop_inst = MachineInst::build_generic(
                            MachineOpcode::Target(TargetInst::X86PopRbp.as_u32()),
                            smallvec::SmallVec::new(),
                        );
                        cursor.emit_before(pop_inst);
                    }

                    cursor.emit_existing_before(inst_id);
                    cursor.remove_current();
                    Ok::<(), crate::error::Error>(())
                })
                .expect("x86_64 prologue/epilogue rewriting should not fail");
        }
    }
}
