use super::{
    inst::{self as generated, TargetInst},
    lowering::{X86_64Lowering, build_x86_copy_inst},
};
use crate::target::{SelectResult, SelectionContext, TargetInstructionSelector};
use veloc_lir::InstRead;
use veloc_lir::MachineOpcode;

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

        let memory = inst.memory();
        let view = inst.view();
        match view {
            veloc_lir::InstView::Return(_) => {
                // ABI result registers stay live through RET, including across
                // otherwise dead instructions moved by the scheduler.
                let inputs = inst.inputs().to_vec();
                let ret = TargetInst::X86Ret.write(
                    ctx.mfunc
                        .editor()
                        .before(ctx.inst_id)
                        .with_effects(&inputs, &[]),
                    &[],
                    &[],
                    [],
                );
                ctx.selected.push(ret);
                return Ok(SelectResult::InPlace);
            }
            veloc_lir::InstView::UnaryReg(copy)
                if copy.opcode == veloc_lir::UnaryRegOpcode::Copy =>
            {
                ctx.selected.push(build_x86_copy_inst(
                    ctx.mfunc.editor().before(ctx.inst_id),
                    copy.dst,
                    copy.src,
                )?);
                return Ok(SelectResult::InPlace);
            }
            _ => {}
        }

        let result = {
            let mut edit = ctx.mfunc.editor();
            let mut insert = edit.before(ctx.inst_id);
            let result = generated::select_instructions(
                &self.lowering,
                features,
                &mut insert,
                ctx.inst_id,
                ctx.selected,
                ctx.edge_transfers,
            )?;
            result
        };

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
