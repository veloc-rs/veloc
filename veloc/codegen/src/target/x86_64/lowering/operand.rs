use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64OperandLowering;

impl X86_64OperandLowering {
    pub fn new(_cpu: CpuDescription) -> Self {
        Self
    }
}

impl TargetOperandLowering for X86_64OperandLowering {
    fn preselect_operand_constraints(
        &self,
        inst: &MachineInst,
        _mfunc: &MachineFunction<PreIselPrepared>,
    ) -> OperandConstraintSet {
        let Some(opcode) = inst.generic_opcode() else {
            return OperandConstraintSet::default();
        };
        generated::generic_inst_metadata(opcode).operand_constraints()
    }

    fn postselect_operand_constraints(
        &self,
        inst: &MachineInst,
        _mfunc: &MachineFunction<SelectedLir>,
    ) -> OperandConstraintSet {
        let MachineOpcode::Target(opcode) = inst.opcode else {
            return OperandConstraintSet::default();
        };
        generated::target_inst_metadata(TargetInst::from_u32(opcode)).operand_constraints()
    }

    fn build_preselect_reg_copy(
        &self,
        mfunc: &MachineFunction<PreIselPrepared>,
        dst: Reg,
        src: Reg,
    ) -> Result<MachineInst, crate::error::Error> {
        Ok(build_x86_copy_inst(mfunc, dst, src).unwrap_or_else(|err| {
            panic!(
                "failed to build x86_64 pre-select reg copy for {:?} <- {:?}: {}",
                dst, src, err
            )
        }))
    }

    fn build_postselect_reg_copy(
        &self,
        mfunc: &MachineFunction<SelectedLir>,
        dst: Reg,
        src: Reg,
    ) -> Result<MachineInst, crate::error::Error> {
        Ok(build_x86_copy_inst(mfunc, dst, src).unwrap_or_else(|err| {
            panic!(
                "failed to build x86_64 post-select reg copy for {:?} <- {:?}: {}",
                dst, src, err
            )
        }))
    }
}
