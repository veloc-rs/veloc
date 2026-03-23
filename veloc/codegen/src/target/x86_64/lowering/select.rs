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

        if matches!(inst.opcode, MachineOpcode::Target(_)) {
            return Ok(SelectResult::Keep);
        }

        if let MachineOpcode::Generic(_opcode) = inst.opcode {
            match _opcode {
                GenericOpcode::G_FCMP => {
                    let fcmp = inst.as_fcmp().unwrap_or_else(|err| {
                        panic!("invalid fcmp instruction during x86_64 selection: {}", err);
                    });
                    if matches!(fcmp.cc, FloatCC::Eq | FloatCC::Ne) {
                        return self.lowering.select_fcmp(ctx, &inst);
                    }
                }
                GenericOpcode::G_SELECT => {
                    let select = inst.as_select().unwrap_or_else(|err| {
                        panic!(
                            "invalid select instruction during x86_64 selection: {}",
                            err
                        );
                    });
                    let dst_ty = if select.dst.is_vreg() {
                        ctx.mfunc.vreg_data(select.dst).ty
                    } else {
                        panic!(
                            "x86_64 select destination must be a virtual register before regalloc",
                        );
                    };
                    if dst_ty.is_float() {
                        return self.lowering.select_select(ctx, &inst);
                    }
                }
                _ => {}
            }
        }

        let result = {
            let selected = core::mem::take(ctx.selected);
            let mut out = selected;
            let x86_ctx = X86SelectionContext { base: ctx, cpu };
            let res = generated::select_instructions(&x86_ctx, &inst, &mut out);
            *ctx.selected = out;
            res.unwrap_or_else(|err| panic!("x86_64 generated selector failed: {}", err))
        };

        Ok(result)
    }
}
