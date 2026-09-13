//! Target facts used by scheduling and allocation, not by generic algorithms.
use super::isle::TargetInst;
use veloc_lir::{InstId, MachineOpcode, MachineOperand, Reg, Writable};
use veloc_mir::{Type, TypeInfo};

pub(super) fn spill_instruction(
    writer: veloc_lir::InstWriter<'_>,
    load: bool,
    reg: Reg,
    base: Reg,
    offset: i64,
    ty: Type,
) -> crate::Result<InstId> {
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
    Ok(writer.generic(
        MachineOpcode::Target(op.as_u32()),
        smallvec::smallvec![
            value,
            MachineOperand::Use(base),
            MachineOperand::Imm(offset)
        ],
    ))
}
