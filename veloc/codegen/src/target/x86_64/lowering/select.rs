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
        let inst = ctx.mfunc.dfg[ctx.inst_id].clone();

        if !inst.is_generic() {
            return Ok(SelectResult::Keep);
        }

        let view = inst.generic_view()?;
        match view {
            veloc_lir::InstView::UnaryReg(copy)
                if copy.opcode == veloc_lir::UnaryRegOpcode::COPY =>
            {
                ctx.selected
                    .push(build_x86_copy_inst(ctx.mfunc, copy.dst, copy.src)?);
                return Ok(SelectResult::InPlace);
            }
            veloc_lir::InstView::FCmp(fcmp) if matches!(fcmp.cc, FloatCC::Eq | FloatCC::Ne) => {
                return self.lowering.select_fcmp(ctx, fcmp);
            }
            veloc_lir::InstView::Select(select) => {
                let dst_ty = if select.dst.is_vreg() {
                    ctx.mfunc.vreg_data(select.dst).ty
                } else {
                    panic!("select destination must be a virtual register before regalloc");
                };
                if dst_ty.is_float() {
                    return self.lowering.select_select(ctx, select);
                }
            }
            _ => {}
        }

        let result = {
            let selected = core::mem::take(ctx.selected);
            let mut out = selected;
            let x86_ctx = X86SelectionContext { base: ctx, cpu };
            let res = generated::select_instructions(&x86_ctx, &inst, &view, &mut out);
            *ctx.selected = out;
            res.unwrap_or_else(|err| panic!("x86_64 generated selector failed: {}", err))
        };

        if matches!(
            inst.generic_opcode(),
            Some(GenericOpcode::G_CALL | GenericOpcode::G_CALLIND)
        ) {
            use crate::target::arch::{AbiLocation, CallConv, TargetArch};
            let sig = &ctx.mfunc.call_info(ctx.inst_id).sig;
            let cc = CallConv::from(sig.call_conv);
            let plan = cc.plan_callsite(TargetArch::X86_64, sig.params(), sig.returns())?;
            let preserved = cc.preserved_regs(TargetArch::X86_64);
            for selected in ctx.selected.iter_mut() {
                if !matches!(selected.opcode, MachineOpcode::Target(op) if op == TargetInst::X86Call.as_u32() || op == TargetInst::X86CallReg.as_u32())
                {
                    continue;
                }
                // Keep ABI register uses/clobbers explicit after G_CALL disappears.
                for part in plan.args.iter().flat_map(|a| &a.parts) {
                    if let AbiLocation::Reg(reg) = part.loc {
                        selected.operands.push(MachineOperand::Use(reg));
                    }
                }
                for reg in generated::PHYS_REG_INFOS {
                    if !preserved.contains(&reg.preg)
                        && reg.preg != generated::REG_RSP
                        && reg.preg != generated::REG_RBP
                    {
                        selected
                            .operands
                            .push(MachineOperand::Def(Writable(reg.preg)));
                    }
                }
            }
        }
        if let Some(access) = inst.memory {
            let mut memory_inst = None;
            for (index, selected) in ctx.selected.iter().enumerate() {
                let MachineOpcode::Target(op) = selected.opcode else {
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
            ctx.selected[index].memory = Some(access);
        }
        Ok(result)
    }
}
