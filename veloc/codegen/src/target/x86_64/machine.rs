//! Target facts used by scheduling and allocation, not by generic algorithms.
use super::inst::TargetInst;
use veloc_lir::{InstId, MachineOpcode, Reg};
use veloc_mir::{Type, TypeInfo};

pub(super) fn spill_instruction(
    writer: veloc_lir::InstWriter<'_>,
    kind: crate::target::SpillKind,
    reg: Reg,
    slot: veloc_lir::StackSlot,
    ty: Type,
) -> crate::Result<InstId> {
    use crate::target::SpillKind::{Load, Store};
    use TargetInst::*;
    let op = match (kind, ty) {
        (Load, Type::F32) => X86LoadF32Stack,
        (Load, Type::F64) => X86LoadF64Stack,
        (Store, Type::F32) => X86StoreF32Stack,
        (Store, Type::F64) => X86StoreF64Stack,
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
                (Load, 1) => X86Load8U32Stack,
                (Load, 2) => X86Load16U32Stack,
                (Load, 4) => X86Load32Stack,
                (Load, 8) => X86Load64Stack,
                (Store, 1) => X86Store8Stack,
                (Store, 2) => X86Store16Stack,
                (Store, 4) => X86Store32Stack,
                (Store, 8) => X86Store64Stack,
                _ => return Err(crate::Error::codegen("unsupported spill width")),
            }
        }
        _ => return Err(crate::Error::codegen("unsupported spill type")),
    };
    let results = if kind == Load { &[reg][..] } else { &[][..] };
    let inputs = if kind == Load { &[][..] } else { &[reg][..] };
    Ok(writer.write(
        MachineOpcode::Target(op.as_u32()),
        results,
        inputs,
        [veloc_lir::FieldValue::StackSlot(slot)],
    ))
}
