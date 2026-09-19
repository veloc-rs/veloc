use super::{
    inst::{self as generated, TargetInst},
    lowering::build_x86_copy_inst,
};
use crate::target::{OperandConstraintSet, TargetOperandLowering};
use veloc_lir::{InstId, MachineFunction, MachineOpcode, Reg};

#[derive(Debug, Clone, Copy)]
pub struct X86_64OperandLowering;

impl TargetOperandLowering for X86_64OperandLowering {
    fn preselect_operand_constraints(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction,
    ) -> OperandConstraintSet {
        let Some(opcode) = inst.generic_opcode() else {
            return OperandConstraintSet::default();
        };
        generated::generic_inst_metadata(opcode).operand_constraints()
    }

    fn postselect_operand_constraints(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction,
    ) -> OperandConstraintSet {
        let MachineOpcode::Target(opcode) = inst.opcode() else {
            return OperandConstraintSet::default();
        };
        generated::target_inst_metadata(TargetInst::from_u32(opcode)).operand_constraints()
    }

    fn build_preselect_reg_copy(
        &self,
        mfunc: &mut MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> Result<InstId, crate::error::Error> {
        Ok(build_x86_copy_inst(mfunc, dst, src).unwrap_or_else(|err| {
            panic!(
                "failed to build x86_64 pre-select reg copy for {:?} <- {:?}: {}",
                dst, src, err
            )
        }))
    }

    fn build_postselect_reg_copy(
        &self,
        mfunc: &mut MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> Result<InstId, crate::error::Error> {
        Ok(build_x86_copy_inst(mfunc, dst, src).unwrap_or_else(|err| {
            panic!(
                "failed to build x86_64 post-select reg copy for {:?} <- {:?}: {}",
                dst, src, err
            )
        }))
    }
}
