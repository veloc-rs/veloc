//! Shared x86 copy construction and selection predicates.
use super::inst::{self as generated, TargetInst};
use veloc_lir::{InstId, MachineFunction, Reg};
use veloc_mir::{Type, TypeInfo};

/// x86_64 专属的 Context 扩展 (架构私有)
pub trait X86LoweringContext {
    fn has_bmi2(&self) -> bool;

    fn has_avx2(&self) -> bool;
}

pub(super) fn x86_mov_opcode_for_type(ty: Type) -> Result<TargetInst, crate::error::Error> {
    if ty == Type::F32 {
        Ok(TargetInst::X86Movss)
    } else if ty == Type::F64 {
        Ok(TargetInst::X86Movsd)
    } else if ty
        .bit_size()
        .and_then(|size| size.fixed_bits())
        .is_some_and(|bits| bits <= 32)
    {
        Ok(TargetInst::X86Mov32)
    } else if ty
        .bit_size()
        .and_then(|size| size.fixed_bits())
        .is_some_and(|bits| bits <= 64)
        || ty.is_ptr()
    {
        Ok(TargetInst::X86Mov64)
    } else {
        panic!("unsupported type for x86_64 move: {:?}", ty);
    }
}

fn x86_copy_type_for_regs(
    mfunc: &MachineFunction,
    dst: Reg,
    src: Reg,
) -> Result<Type, crate::error::Error> {
    if dst.is_vreg() {
        return Ok(mfunc.vreg_data(dst).ty);
    }
    if src.is_vreg() {
        return Ok(mfunc.vreg_data(src).ty);
    }
    panic!(
        "cannot infer x86 copy type from physical registers {:?} <- {:?}",
        dst, src
    )
}

pub(super) fn build_x86_copy_inst(
    mut insert: veloc_lir::InstInserter<'_>,
    dst: Reg,
    src: Reg,
) -> Result<InstId, crate::error::Error> {
    let ty = x86_copy_type_for_regs(&insert, dst, src)?;
    let opcode = x86_mov_opcode_for_type(ty)?;
    Ok(opcode.write(insert.writer(), &[dst], &[src], []))
}

impl X86LoweringContext for generated::FeatureSet {
    fn has_bmi2(&self) -> bool {
        self.contains(generated::Feature::BMI2)
    }
    fn has_avx2(&self) -> bool {
        self.contains(generated::Feature::AVX2)
    }
}

impl crate::isel::SelectHooks for generated::FeatureSet {
    fn predicate(&self, id: u32, reg: Reg) -> bool {
        generated::selection_predicate(self, id, reg)
    }
}
