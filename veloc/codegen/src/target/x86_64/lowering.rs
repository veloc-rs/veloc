//! Shared x86 copy construction.
use super::inst::TargetInst;
use veloc_lir::Reg;
use veloc_mir::Type;

pub(super) fn copy_opcode(
    desc: &crate::target::TargetDescription,
    dst: Reg,
    src: Reg,
    ty: Type,
) -> crate::Result<TargetInst> {
    use TargetInst::*;
    let dst = desc.scalar_storage_type(dst, ty)?;
    let src = desc.scalar_storage_type(src, ty)?;
    Ok(match (dst, src) {
        (Type::F32, Type::F32) => X86Movss,
        (Type::F64, Type::F64) => X86Movsd,
        (Type::F32, Type::I32) => X86MovdToXmm,
        (Type::F64, Type::I64) => X86MovqToXmm,
        (Type::I32, Type::F32) => X86MovdFromXmm,
        (Type::I64, Type::F64) => X86MovqFromXmm,
        (Type::I64, Type::I64) => X86Mov64,
        (Type::I8 | Type::I16 | Type::I32, Type::I8 | Type::I16 | Type::I32) => X86Mov32,
        _ => {
            return Err(crate::Error::codegen(
                "unsupported scalar register transfer",
            ));
        }
    })
}
