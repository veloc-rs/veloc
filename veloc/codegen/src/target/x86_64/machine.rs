//! Target facts used by scheduling and allocation, not by generic algorithms.
use super::isle::TargetInst;
use crate::target::arch::ScheduleInfo;
use veloc_lir::{MachineInst, MachineOpcode, MachineOperand, Reg, Writable};
use veloc_mir::Type;

pub(super) fn schedule_info(inst: &MachineInst) -> Option<ScheduleInfo> {
    if inst.memory.is_some() {
        return None;
    }
    let MachineOpcode::Target(op) = inst.opcode else {
        return None;
    };
    super::isle::target_inst_metadata(TargetInst::from_u32(op)).schedule
}

pub(super) fn spill_instruction(
    load: bool,
    reg: Reg,
    base: Reg,
    offset: i64,
    ty: Type,
) -> crate::Result<MachineInst> {
    use TargetInst::*;
    let op = match (load, ty) {
        (true, Type::F32) => X86LoadF32,
        (true, Type::F64) => X86LoadF64,
        (false, Type::F32) => X86StoreF32,
        (false, Type::F64) => X86StoreF64,
        _ if ty.is_ptr() || (ty.is_scalar() && (ty.is_integer() || ty == Type::BOOL)) => {
            match (
                load,
                if ty.is_ptr() {
                    8
                } else {
                    ty.fixed_size_bytes().unwrap()
                },
            ) {
                (true, 1) => X86Load8U32,
                (true, 2) => X86Load16U32,
                (true, 4) => X86Load32,
                (true, 8) => X86Load64,
                (false, 1) => X86Store8,
                (false, 2) => X86Store16,
                (false, 4) => X86Store32,
                (false, 8) => X86Store64,
                _ => return Err(crate::Error::codegen("unsupported spill width")),
            }
        }
        _ => return Err(crate::Error::codegen("unsupported spill type")),
    };
    let value = if load {
        MachineOperand::Def(Writable(reg))
    } else {
        MachineOperand::Use(reg)
    };
    Ok(MachineInst::build_generic(
        MachineOpcode::Target(op.as_u32()),
        smallvec::smallvec![
            value,
            MachineOperand::Use(base),
            MachineOperand::Imm(offset)
        ],
    ))
}
