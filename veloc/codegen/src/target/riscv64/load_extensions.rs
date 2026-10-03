//! Fuse a load's sole sign-extension user at the load's original position.
//! Unlike pure expression matching, this never duplicates or moves an access.
use super::inst::TargetInst;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use veloc_lir::{FieldValueRef, Fields, MachineOpcode};

pub(super) struct FoldLoadExtensions;

impl FunctionPass for FoldLoadExtensions {
    fn name(&self) -> &'static str {
        "fold-load-extensions"
    }
    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }

    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        use TargetInst::*;
        let consumers: Vec<_> = cx
            .function()
            .blocks()
            .flat_map(|b| cx.function().block_insts(b))
            .collect();
        let mut f = cx.edit();
        for consumer in consumers {
            let MachineOpcode::Target(op) = f.inst(consumer).opcode() else {
                continue;
            };
            let (load, replacement) = match TargetInst::from_u32(op) {
                RvSext8 | RvSext8Zbb => (RvLoad8, RvLoad8Signed),
                RvSext16 | RvSext16Zbb => (RvLoad16, RvLoad16Signed),
                RvSext32 => (RvLoad32, RvLoad32),
                _ => continue,
            };
            let input = f.inst(consumer).inputs()[0];
            let result = f.inst(consumer).results()[0];
            if !input.is_vreg() || !result.is_vreg() {
                continue;
            }
            let Some(def) = f.defs(input).single() else {
                continue;
            };
            let producer = def.inst();
            if f.inst(producer).opcode() != MachineOpcode::Target(load.as_u32()) {
                continue;
            }
            let mut uses = f.uses(input);
            if uses.next().is_none_or(|site| site.inst() != consumer) || uses.next().is_some() {
                continue;
            }
            let source = f.inst(producer);
            let base = source.inputs()[0];
            let FieldValueRef::Imm(&offset) = source.fields().read(0) else {
                unreachable!()
            };
            let flags = source.mem_flags().expect("load flags");
            replacement.write(
                f.editor().replace(producer),
                &[result],
                &[base],
                Fields::Memory { offset, flags },
            );
            f.editor().invalidate_inst(consumer);
        }
        Ok(())
    }
}
