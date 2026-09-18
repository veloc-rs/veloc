//! Target facts used by scheduling and allocation, not by generic algorithms.
use super::inst::TargetInst;
use veloc_lir::{InstField, InstId, MachineOpcode, Reg};
use veloc_mir::{Type, TypeInfo};

pub(super) fn spill_instruction(
    writer: veloc_lir::InstWriter<'_>,
    kind: crate::target::arch::SpillKind,
    reg: Reg,
    base: Reg,
    offset: i64,
    ty: Type,
) -> crate::Result<InstId> {
    use crate::target::arch::SpillKind::{Load, Store};
    use TargetInst::*;
    let op = match (kind, ty) {
        (Load, Type::F32) => X86LoadF32,
        (Load, Type::F64) => X86LoadF64,
        (Store, Type::F32) => X86StoreF32,
        (Store, Type::F64) => X86StoreF64,
        _ if ty.is_ptr() || (ty.is_scalar() && (ty.is_integer() || ty == Type::BOOL)) => {
            match (
                kind,
                if ty.is_ptr() {
                    8
                } else {
                    super::DATA_LAYOUT
                        .layout_of(ty)
                        .and_then(|layout| layout.store_size.fixed_bytes())
                        .ok_or_else(|| crate::Error::codegen("unknown spill layout"))?
                },
            ) {
                (Load, 1) => X86Load8U32,
                (Load, 2) => X86Load16U32,
                (Load, 4) => X86Load32,
                (Load, 8) => X86Load64,
                (Store, 1) => X86Store8,
                (Store, 2) => X86Store16,
                (Store, 4) => X86Store32,
                (Store, 8) => X86Store64,
                _ => return Err(crate::Error::codegen("unsupported spill width")),
            }
        }
        _ => return Err(crate::Error::codegen("unsupported spill type")),
    };
    let results = if kind == Load { &[reg][..] } else { &[][..] };
    let inputs = if kind == Load {
        &[base][..]
    } else {
        &[reg, base][..]
    };
    Ok(writer.write(
        MachineOpcode::Target(op.as_u32()),
        results,
        inputs,
        &[InstField::Imm(offset)],
    ))
}
