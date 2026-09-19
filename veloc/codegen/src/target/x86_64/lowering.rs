//! Shared x86 copy construction and selection predicates.
use super::inst::{self as generated, TargetInst};
use veloc_lir::{InstField, InstId, MachineFunction, Reg};
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
    mfunc: &mut MachineFunction,
    dst: Reg,
    src: Reg,
) -> Result<InstId, crate::error::Error> {
    let ty = x86_copy_type_for_regs(mfunc, dst, src)?;
    let opcode = x86_mov_opcode_for_type(ty)?;
    Ok(opcode.write(mfunc.editor().writer(), &[dst], &[src], &[]))
}

pub(super) fn build_target_inst(
    writer: veloc_lir::InstWriter<'_>,
    opcode: TargetInst,
    results: &[Reg],
    inputs: &[Reg],
    fields: &[InstField],
) -> InstId {
    opcode.write(writer, results, inputs, fields)
}

/// x86_64 后端共享 lowering helper。
#[derive(Debug, Clone, Copy)]
pub struct X86_64Lowering {
    /// 当前 target instance 选中的 CPU 描述。
    pub features: generated::FeatureSet,
}

impl X86_64Lowering {
    pub fn new(features: generated::FeatureSet) -> Self {
        Self { features }
    }
}

impl X86LoweringContext for X86_64Lowering {
    fn has_bmi2(&self) -> bool {
        self.features.contains(generated::Feature::BMI2)
    }
    fn has_avx2(&self) -> bool {
        self.features.contains(generated::Feature::AVX2)
    }
}
