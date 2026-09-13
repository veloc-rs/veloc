use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64Selector {
    pub(super) lowering: X86_64Lowering,
}

impl X86_64Selector {
    pub fn new(cpu: CpuDescription) -> Self {
        Self {
            lowering: X86_64Lowering::new(cpu),
        }
    }
}

impl TargetInstructionSelector for X86_64Selector {
    fn select_instruction(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
    ) -> Result<SelectResult, crate::error::Error> {
        let cpu = self.lowering.cpu;
        let inst = ctx.mfunc.inst(ctx.inst_id);

        if !inst.is_generic() {
            return Ok(SelectResult::Keep);
        }

        let opcode = inst.opcode();
        let memory = inst.memory();
        let view = inst.generic_view()?;
        match view {
            veloc_lir::InstView::Return(_) => {
                // ABI result registers stay live through RET, including across
                // otherwise dead instructions moved by the scheduler.
                let operands = inst.operands().iter().cloned().collect();
                ctx.selected.push(build_target_inst(
                    ctx.mfunc.writer(),
                    TargetInst::X86Ret,
                    operands,
                ));
                return Ok(SelectResult::InPlace);
            }
            veloc_lir::InstView::UnaryReg(copy)
                if copy.opcode == veloc_lir::UnaryRegOpcode::COPY =>
            {
                ctx.selected
                    .push(build_x86_copy_inst(ctx.mfunc, copy.dst, copy.src)?);
                return Ok(SelectResult::InPlace);
            }
            veloc_lir::InstView::FCmp(fcmp)
                if matches!(
                    fcmp.cc,
                    FloatCC::Eq | FloatCC::Ne | FloatCC::Lt | FloatCC::Le
                ) =>
            {
                return self.lowering.select_fcmp(ctx, fcmp);
            }
            veloc_lir::InstView::Select(select) => {
                return self.lowering.select_select(ctx, select);
            }
            _ => {}
        }

        let result = {
            let (vregs, store) = ctx.mfunc.instruction_parts();
            let mut x86_ctx = X86SelectionContext { vregs, cpu };
            generated::select_instructions(&mut x86_ctx, store, ctx.inst_id, ctx.selected)?
        };

        if matches!(
            opcode,
            MachineOpcode::Generic(GenericOpcode::G_CALL | GenericOpcode::G_CALLIND)
        ) {
            use crate::target::arch::{AbiLocation, CallConv, TargetArch};
            let sig = &ctx.mfunc.call_info(ctx.inst_id).sig;
            let cc = CallConv::from(sig.call_conv);
            let plan = cc.plan_callsite(TargetArch::X86_64, sig.params(), sig.returns())?;
            let preserved = cc.preserved_regs(TargetArch::X86_64);
            for &selected in ctx.selected.iter() {
                if !matches!(ctx.mfunc.inst(selected).opcode(), MachineOpcode::Target(op) if op == TargetInst::X86Call.as_u32() || op == TargetInst::X86CallReg.as_u32())
                {
                    continue;
                }
                let mut operands = ctx.mfunc.inst(selected).operands().to_vec();
                // Keep ABI register uses/clobbers explicit after G_CALL disappears.
                for part in plan.args.iter().flat_map(|a| &a.parts) {
                    if let AbiLocation::Reg(reg) = part.loc {
                        operands.push(MachineOperand::Use(reg));
                    }
                }
                for reg in generated::PHYS_REG_INFOS {
                    if !preserved.contains(&reg.preg)
                        && reg.preg != generated::REG_RSP
                        && reg.preg != generated::REG_RBP
                    {
                        operands.push(MachineOperand::Def(Writable(reg.preg)));
                    }
                }
                ctx.mfunc.set_inst_operands(selected, operands);
            }
        }
        // A selected conditional is a branch followed by a jump. Each keeps
        // its own edge arguments, including duplicate targets with different args.
        let edge_args = match ctx.mfunc.inst_extra(ctx.inst_id) {
            Some(InstExtra::Branch(info)) => alloc::vec![info.args.clone()],
            Some(InstExtra::BranchCond(info)) => {
                alloc::vec![info.then_args.clone(), info.else_args.clone()]
            }
            _ => Vec::new(),
        };
        if !edge_args.is_empty() {
            let branches: Vec<_> = ctx
                .selected
                .iter()
                .copied()
                .filter(|&id| {
                    let MachineOpcode::Target(op) = ctx.mfunc.inst(id).opcode() else {
                        return false;
                    };
                    matches!(
                        generated::target_inst_metadata(TargetInst::from_u32(op)).flow,
                        veloc_lir::ControlFlow::Branch | veloc_lir::ControlFlow::Jump
                    )
                })
                .collect();
            if branches.len() != edge_args.len() {
                return Err(crate::Error::codegen(
                    "selection changed the number of outgoing edges",
                ));
            }
            for (id, args) in branches.into_iter().zip(edge_args) {
                ctx.mfunc
                    .set_inst_extra(id, InstExtra::Branch(veloc_lir::BranchInfo { args }));
            }
        }
        if let Some(access) = memory {
            let mut memory_inst = None;
            for (index, &selected) in ctx.selected.iter().enumerate() {
                let MachineOpcode::Target(op) = ctx.mfunc.inst(selected).opcode() else {
                    continue;
                };
                if let Some(shape) =
                    generated::target_inst_metadata(TargetInst::from_u32(op)).memory
                {
                    if shape != (access.kind, access.bytes) || memory_inst.replace(index).is_some()
                    {
                        return Err(crate::error::Error::codegen(
                            "selection changed the memory access direction, size or count",
                        ));
                    }
                }
            }
            let index = memory_inst.ok_or_else(|| {
                crate::error::Error::codegen("selection dropped the source memory access")
            })?;
            ctx.mfunc.set_inst_memory(ctx.selected[index], Some(access));
        }
        Ok(result)
    }
}
