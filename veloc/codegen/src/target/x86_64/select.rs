use super::{
    inst::{self as generated, TargetInst},
    lowering::{X86_64Lowering, build_x86_copy_inst},
};
use crate::target::{SelectResult, SelectionContext, TargetInstructionSelector};
use veloc_lir::InstRead;
use veloc_lir::{GenericOpcode, MachineOpcode};

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
                let ret = TargetInst::X86Ret.write(ctx.mfunc.editor().writer(), &[], &[], []);
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
            let (mut vregs, mut store) = edit.instruction_parts();
            let result = generated::select_instructions(
                &self.lowering,
                &mut vregs,
                features,
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
            // ABI lowering owns the dynamic call contract. Selection only
            // combines it with the selected opcode's static register effects.
            let source = ctx
                .mfunc
                .inst(ctx.inst_id)
                .effects()
                .map(|e| veloc_lir::RegEffects {
                    uses: e.uses.to_vec(),
                    defs: e.defs.to_vec(),
                })
                .expect("call must be ABI lowered before selection");
            let calls: alloc::vec::Vec<_> = ctx.selected.iter().copied().filter(|&id| {
                matches!(ctx.mfunc.inst(id).opcode(), MachineOpcode::Target(op)
                    if generated::target_inst_metadata(TargetInst::from_u32(op)).flow == veloc_lir::ControlFlow::Call)
            }).collect();
            assert_eq!(calls.len(), 1, "selection must preserve one call boundary");
            let selected = calls[0];
            let mut effects = source;
            if let Some(existing) = ctx.mfunc.inst(selected).effects() {
                effects.uses.extend_from_slice(existing.uses);
                effects.defs.extend_from_slice(existing.defs);
            }
            effects.uses.sort_unstable();
            effects.uses.dedup();
            effects.defs.sort_unstable();
            effects.defs.dedup();
            ctx.mfunc.editor().set_inst_effects(selected, effects);
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
