use super::*;
use veloc_lir::InstRead;

#[derive(Debug, Clone, Copy)]
pub struct X86_64Selector {
    pub(super) lowering: X86_64Lowering,
}

impl X86_64Selector {
    pub fn new(features: generated::FeatureSet) -> Self {
        Self {
            lowering: X86_64Lowering::new(features),
        }
    }
}

impl TargetInstructionSelector for X86_64Selector {
    fn select_instruction(
        &self,
        ctx: &mut SelectionContext<'_>,
    ) -> Result<SelectResult, crate::error::Error> {
        let features = self.lowering.features;
        let inst = ctx.mfunc.inst(ctx.inst_id);

        if !inst.is_generic() {
            return Ok(SelectResult::Keep);
        }

        let opcode = inst.opcode();
        let memory = inst.memory();
        let view = inst.view();
        match view {
            veloc_lir::InstView::Return(_) => {
                // ABI result registers stay live through RET, including across
                // otherwise dead instructions moved by the scheduler.
                let inputs = inst.inputs().to_vec();
                let ret = build_target_inst(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86Ret,
                    &[],
                    &[],
                    &[],
                );
                ctx.mfunc.editor().set_inst_effects(
                    ret,
                    veloc_lir::RegEffects {
                        uses: inputs,
                        ..Default::default()
                    },
                );
                ctx.selected.push(ret);
                return Ok(SelectResult::InPlace);
            }
            veloc_lir::InstView::UnaryReg(copy)
                if copy.opcode == veloc_lir::UnaryRegOpcode::Copy =>
            {
                ctx.selected
                    .push(build_x86_copy_inst(ctx.mfunc, copy.dst, copy.src)?);
                return Ok(SelectResult::InPlace);
            }
            _ => {}
        }

        let result = {
            let mut edit = ctx.mfunc.editor();
            let (vregs, mut store) = edit.instruction_parts();
            let mut x86_ctx = X86SelectionContext { vregs, features };
            let result = generated::select_instructions(
                &mut x86_ctx,
                &mut store,
                ctx.inst_id,
                ctx.selected,
            )?;
            ctx.edge_transfers.extend(store.into_edge_transfers());
            result
        };

        if matches!(
            opcode,
            MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind)
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
                let mut effects = ctx
                    .mfunc
                    .inst(selected)
                    .effects()
                    .map(|e| veloc_lir::RegEffects {
                        uses: e.uses.to_vec(),
                        defs: e.defs.to_vec(),
                    })
                    .unwrap_or_default();
                // Keep ABI register uses/clobbers explicit after Call disappears.
                for part in plan.args.iter().flat_map(|a| &a.parts) {
                    if let AbiLocation::Reg(reg) = part.loc {
                        effects.uses.push(reg);
                    }
                }
                for reg in generated::PHYS_REG_INFOS {
                    if !preserved.contains(&reg.preg)
                        && reg.preg != generated::REG_RSP
                        && reg.preg != generated::REG_RBP
                    {
                        effects.defs.push(reg.preg);
                    }
                }
                effects.uses.sort_unstable();
                effects.uses.dedup();
                effects.defs.sort_unstable();
                effects.defs.dedup();
                ctx.mfunc.editor().set_inst_effects(selected, effects);
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
            ctx.mfunc
                .editor()
                .set_inst_memory(ctx.selected[index], Some(access));
        }
        Ok(result)
    }
}
