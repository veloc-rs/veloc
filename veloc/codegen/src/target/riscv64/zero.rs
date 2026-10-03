//! Use the architectural zero register for unconstrained instruction operands.
//! Edge arguments retain their typed SSA values until allocation.
use super::inst::TargetInst;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use std::collections::HashSet;
use veloc_lir::{FieldValueRef, MachineOpcode, Reg};

pub(super) struct ZeroOperands;
impl FunctionPass for ZeroOperands {
    fn name(&self) -> &'static str {
        "zero-operands"
    }
    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let insts: Vec<_> = cx
            .function()
            .blocks()
            .flat_map(|b| cx.function().block_insts(b))
            .collect();
        let mut zeros = HashSet::new();
        let mut definitions = Vec::new();
        for &inst in &insts {
            let view = cx.function().inst(inst);
            if matches!(view.opcode(), MachineOpcode::Target(op) if matches!(TargetInst::from_u32(op), TargetInst::RvLi32 | TargetInst::RvLi64))
                && matches!(view.fields().read(0), FieldValueRef::Imm(&0))
            {
                zeros.insert(view.results()[0]);
                definitions.push(inst);
            }
        }
        let target = cx.target;
        let mut f = cx.edit();
        for inst in insts {
            let view = f.inst(inst);
            let MachineOpcode::Target(op) = view.opcode() else {
                continue;
            };
            // ABI and tied operands may require writable registers.
            if view.constraints().len() != 0
                || !target.instruction_metadata(op).constraints.is_empty()
            {
                continue;
            }
            let inputs: Vec<_> = view
                .inputs()
                .iter()
                .enumerate()
                .filter(|(_, r)| zeros.contains(r))
                .map(|(i, _)| i)
                .collect();
            for index in inputs {
                f.editor().set_inst_input(inst, index, Reg::new_preg(0));
            }
        }
        for inst in definitions {
            let reg = f.inst(inst).results()[0];
            if f.uses(reg).next().is_none() {
                f.editor().invalidate_inst(inst);
            }
        }
        Ok(())
    }
}
