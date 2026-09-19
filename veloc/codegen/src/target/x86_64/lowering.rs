//! x86_64 Target Lowering
//!
//! 使用 Spec 生成的代码
//! 进行指令选择。

mod frame;
mod legalize;
mod operand;
mod pass_config;
mod select;

pub use crate::isel::SelectResult;
use crate::passes::lowering::{LegalizeAction, RewriteContext};
use crate::target::arch::{
    CallConv as TargetCallConv, LoweringContext, OperandConstraintSet, SelectionContext,
    TargetArch, TargetFrameLowering, TargetInstructionSelector, TargetLegalizer,
    TargetOperandLowering, TargetPassConfig, TargetPostIsel,
};
use crate::target::x86_64::inst::{self as generated, TargetInst};
use alloc::vec::Vec;
pub use frame::X86_64FrameLowering;
pub use legalize::X86_64Legalizer;
pub use operand::X86_64OperandLowering;
pub use pass_config::{X86_64PassConfig, X86_64PostIsel};
pub use select::X86_64Selector;
use veloc_lir::{
    GenericOpcode, InstField, InstId, MachineFunction, MachineOpcode, Reg, VReg, Writable,
};
use veloc_mir::{Type, TypeInfo};

/// x86_64 专属的 Context 扩展 (架构私有)
pub trait X86LoweringContext: LoweringContext {
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

fn build_x86_copy_inst(
    mfunc: &mut MachineFunction,
    dst: Reg,
    src: Reg,
) -> Result<InstId, crate::error::Error> {
    let ty = x86_copy_type_for_regs(mfunc, dst, src)?;
    let opcode = x86_mov_opcode_for_type(ty)?;
    Ok(opcode.write(mfunc.editor().writer(), &[dst], &[src], &[]))
}

fn build_target_inst(
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

/// x86_64 专属的 Context 扩展实现
pub struct X86SelectionContext<'a> {
    pub vregs: veloc_lir::VRegBuilder<'a>,
    pub features: generated::FeatureSet,
}
impl LoweringContext for X86SelectionContext<'_> {
    fn alloc_tmp(&mut self, ty: Type) -> Reg {
        self.vregs.alloc(veloc_lir::VRegData { ty, bank: None })
    }
    fn get_type(&self, vreg: VReg) -> Type {
        self.vregs.get(vreg).ty
    }
    fn get_vreg(&self, inst: &veloc_lir::InstRef<'_>, index: usize) -> Option<VReg> {
        let reg = inst.inputs().get(index)?;
        reg.is_vreg().then(|| VReg::from_u32(reg.index()))
    }
}
impl X86LoweringContext for X86SelectionContext<'_> {
    fn has_bmi2(&self) -> bool {
        self.features.contains(generated::Feature::BMI2)
    }
    fn has_avx2(&self) -> bool {
        self.features.contains(generated::Feature::AVX2)
    }
}

impl crate::target::arch::TargetFeatures for X86SelectionContext<'_> {
    type Features = generated::FeatureSet;
    fn supports_features(&self, required: Self::Features) -> bool {
        self.features.contains_all(required)
    }
}
