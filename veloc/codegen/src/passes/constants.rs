//! Fold representation changes introduced by legalization before matching.
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use veloc_lir::{GenericOpcode, InstBuild, InstRead, InstView};
use veloc_mir::Int;

pub struct FoldConstantCasts;

impl FunctionPass for FoldConstantCasts {
    fn name(&self) -> &'static str {
        "fold-constant-casts"
    }
    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Legal
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let mut f = cx.edit();
        let insts: Vec<_> = f.blocks().flat_map(|b| f.block_insts(b)).collect();
        loop {
            let mut changed = false;
            for &inst in &insts {
                let Some(op @ (GenericOpcode::Trunc | GenericOpcode::Zext | GenericOpcode::Sext)) =
                    f.inst(inst).generic_opcode()
                else {
                    continue;
                };
                let source = f.inst(inst).inputs()[0];
                let result = f.inst(inst).results()[0];
                let Some(def) = f.defs(source).single() else {
                    continue;
                };
                let InstView::Constant(c) = f.inst(def.inst()).view() else {
                    continue;
                };
                let Some(input) = Int::from_bits(f.vreg_data(source).ty, c.imm as u64) else {
                    continue;
                };
                let bits = if op == GenericOpcode::Sext {
                    input.signed() as u64
                } else {
                    input.to_bits()
                };
                let Some(output) = Int::from_bits(f.vreg_data(result).ty, bits) else {
                    continue;
                };
                f.editor().replace(inst).constant(result, output.signed());
                changed = true;
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }
}
