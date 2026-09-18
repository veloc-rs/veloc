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
use veloc_lir::RegisterBank;
use veloc_lir::{
    GenericOpcode, InstExtra, InstField, InstId, MachineFunction, MachineOpcode, Reg, VReg,
    Writable,
};
use veloc_mir::{FloatCC, Type, TypeInfo};

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

fn build_target_imm(
    writer: veloc_lir::InstWriter<'_>,
    opcode: TargetInst,
    dst: Writable<Reg>,
    imm: i64,
) -> InstId {
    build_target_inst(
        writer,
        opcode,
        &[(dst).to_reg()],
        &[],
        &[InstField::Imm(imm)],
    )
}

fn build_target_unary(
    writer: veloc_lir::InstWriter<'_>,
    opcode: TargetInst,
    dst: Writable<Reg>,
    src: Reg,
) -> InstId {
    opcode.write(writer, &[dst.to_reg()], &[src], &[])
}

fn build_target_binary_uses(
    writer: veloc_lir::InstWriter<'_>,
    opcode: TargetInst,
    lhs: Reg,
    rhs: Reg,
) -> InstId {
    build_target_inst(writer, opcode, &[], &[lhs, rhs], &[])
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

    fn alloc_gpr_temp(&self, mfunc: &mut MachineFunction, ty: Type) -> Reg {
        mfunc.editor().alloc_vreg_in_bank(ty, RegisterBank::GPR)
    }

    fn normalize_cond_to_i32(
        &self,
        ctx: &mut SelectionContext<'_>,
        cond: Reg,
        cond_ty: Type,
    ) -> Reg {
        let test_opcode = if cond_ty
            .bit_size()
            .and_then(|size| size.fixed_bits())
            .is_some_and(|bits| bits <= 32)
        {
            TargetInst::X86Test32
        } else {
            TargetInst::X86Test64
        };
        let cond_byte = self.alloc_gpr_temp(ctx.mfunc, Type::I8);
        let cond_i32 = self.alloc_gpr_temp(ctx.mfunc, Type::I32);

        ctx.selected.push(build_target_binary_uses(
            ctx.mfunc.editor().writer(),
            test_opcode,
            cond,
            cond,
        ));
        ctx.selected.push(build_target_inst(
            ctx.mfunc.editor().writer(),
            TargetInst::X86Setne,
            &[(Writable(cond_byte)).to_reg()],
            &[],
            &[],
        ));
        ctx.selected.push(build_target_unary(
            ctx.mfunc.editor().writer(),
            TargetInst::X86Movzx8to32,
            Writable(cond_i32),
            cond_byte,
        ));

        cond_i32
    }

    fn emit_select_i32(
        &self,
        ctx: &mut SelectionContext<'_>,
        dst: Reg,
        cond: Reg,
        true_val: Reg,
        false_val: Reg,
    ) {
        self.emit_select_bits(ctx, dst, cond, true_val, false_val, Type::I32);
    }

    fn emit_select_i64_like(
        &self,
        ctx: &mut SelectionContext<'_>,
        dst: Reg,
        cond: Reg,
        true_val: Reg,
        false_val: Reg,
        ty: Type,
    ) {
        let wide = self.alloc_gpr_temp(ctx.mfunc, Type::I64);
        ctx.selected.push(TargetInst::X86Mov32.write(
            ctx.mfunc.editor().writer(),
            &[wide],
            &[cond],
            &[],
        ));
        self.emit_select_bits(ctx, dst, wide, true_val, false_val, ty);
    }

    fn emit_select_bits(
        &self,
        ctx: &mut SelectionContext<'_>,
        dst: Reg,
        cond: Reg,
        true_val: Reg,
        false_val: Reg,
        ty: Type,
    ) {
        let (mov, sub, xor, and) = if ty == Type::I32 {
            (
                TargetInst::X86Mov32Imm,
                TargetInst::X86Sub32,
                TargetInst::X86Xor32,
                TargetInst::X86And32,
            )
        } else {
            (
                TargetInst::X86Mov64Imm32,
                TargetInst::X86Sub64,
                TargetInst::X86Xor64,
                TargetInst::X86And64,
            )
        };
        let zero = self.alloc_gpr_temp(ctx.mfunc, ty);
        let mask = self.alloc_gpr_temp(ctx.mfunc, ty);
        let diff = self.alloc_gpr_temp(ctx.mfunc, ty);
        let masked = self.alloc_gpr_temp(ctx.mfunc, ty);
        ctx.selected.push(build_target_imm(
            ctx.mfunc.editor().writer(),
            mov,
            Writable(zero),
            0,
        ));
        // false ^ ((true ^ false) & -cond), with a distinct value at each step.
        for (op, output, lhs, rhs) in [
            (sub, mask, zero, cond),
            (xor, diff, true_val, false_val),
            (and, masked, diff, mask),
            (xor, dst, false_val, masked),
        ] {
            ctx.selected
                .push(op.write(ctx.mfunc.editor().writer(), &[output], &[rhs, lhs], &[]));
        }
    }

    fn select_fcmp(
        &self,
        ctx: &mut SelectionContext<'_>,
        fcmp: veloc_lir::FCmpInst,
    ) -> Result<SelectResult, crate::error::Error> {
        let compare_opcode = match if fcmp.lhs.is_vreg() {
            ctx.mfunc.vreg_data(fcmp.lhs).ty
        } else {
            Type::F64
        } {
            Type::F32 => TargetInst::X86Ucomiss,
            Type::F64 => TargetInst::X86Ucomisd,
            other => {
                panic!("unsupported x86_64 fcmp type: {:?}", other);
            }
        };

        ctx.selected.push(build_target_binary_uses(
            ctx.mfunc.editor().writer(),
            compare_opcode,
            fcmp.lhs,
            fcmp.rhs,
        ));

        let emit_setcc_i32 = |ctx: &mut SelectionContext<'_>, opcode: TargetInst| -> Reg {
            let tmp8 = self.alloc_gpr_temp(ctx.mfunc, Type::I8);
            let tmp32 = self.alloc_gpr_temp(ctx.mfunc, Type::I32);
            ctx.selected.push(build_target_inst(
                ctx.mfunc.editor().writer(),
                opcode,
                &[(Writable(tmp8)).to_reg()],
                &[],
                &[],
            ));
            ctx.selected.push(build_target_unary(
                ctx.mfunc.editor().writer(),
                TargetInst::X86Movzx8to32,
                Writable(tmp32),
                tmp8,
            ));
            tmp32
        };

        match fcmp.cc {
            FloatCC::Eq | FloatCC::Lt | FloatCC::Le => {
                let predicate = match fcmp.cc {
                    FloatCC::Eq => TargetInst::X86Sete,
                    FloatCC::Lt => TargetInst::X86Setb,
                    FloatCC::Le => TargetInst::X86Setbe,
                    _ => unreachable!(),
                };
                let is_eq = emit_setcc_i32(ctx, predicate);
                let ordered = emit_setcc_i32(ctx, TargetInst::X86Setnp);
                ctx.selected.push(TargetInst::X86And32.write(
                    ctx.mfunc.editor().writer(),
                    &[fcmp.dst],
                    &[ordered, is_eq],
                    &[],
                ));
            }
            FloatCC::Ne => {
                let is_ne = emit_setcc_i32(ctx, TargetInst::X86Setne);
                let unordered = emit_setcc_i32(ctx, TargetInst::X86Setp);
                ctx.selected.push(TargetInst::X86Or32.write(
                    ctx.mfunc.editor().writer(),
                    &[fcmp.dst],
                    &[unordered, is_ne],
                    &[],
                ));
            }
            other => {
                panic!(
                    "x86_64 ordered/unordered FCMP selector received {:?}",
                    other
                );
            }
        }

        Ok(SelectResult::Replace)
    }

    fn select_select(
        &self,
        ctx: &mut SelectionContext<'_>,
        select: veloc_lir::SelectInst,
    ) -> Result<SelectResult, crate::error::Error> {
        let dst_ty = if select.dst.is_vreg() {
            ctx.mfunc.vreg_data(select.dst).ty
        } else {
            panic!("x86_64 select destination must be a virtual register before regalloc",);
        };
        let cond_ty = if select.cond.is_vreg() {
            ctx.mfunc.vreg_data(select.cond).ty
        } else {
            Type::I64
        };
        let cond_i32 = self.normalize_cond_to_i32(ctx, select.cond, cond_ty);

        match dst_ty {
            Type::F32 => {
                let true_bits = self.alloc_gpr_temp(ctx.mfunc, Type::I32);
                let false_bits = self.alloc_gpr_temp(ctx.mfunc, Type::I32);
                let dst_bits = self.alloc_gpr_temp(ctx.mfunc, Type::I32);

                ctx.selected.push(build_target_unary(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86MovdFromXmm,
                    Writable(true_bits),
                    select.v1,
                ));
                ctx.selected.push(build_target_unary(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86MovdFromXmm,
                    Writable(false_bits),
                    select.v2,
                ));
                self.emit_select_i32(ctx, dst_bits, cond_i32, true_bits, false_bits);
                ctx.selected.push(build_target_unary(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86MovdToXmm,
                    Writable(select.dst),
                    dst_bits,
                ));
            }
            Type::F64 => {
                let true_bits = self.alloc_gpr_temp(ctx.mfunc, Type::I64);
                let false_bits = self.alloc_gpr_temp(ctx.mfunc, Type::I64);
                let dst_bits = self.alloc_gpr_temp(ctx.mfunc, Type::I64);

                ctx.selected.push(build_target_unary(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86MovqFromXmm,
                    Writable(true_bits),
                    select.v1,
                ));
                ctx.selected.push(build_target_unary(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86MovqFromXmm,
                    Writable(false_bits),
                    select.v2,
                ));
                self.emit_select_i64_like(
                    ctx,
                    dst_bits,
                    cond_i32,
                    true_bits,
                    false_bits,
                    Type::I64,
                );
                ctx.selected.push(build_target_unary(
                    ctx.mfunc.editor().writer(),
                    TargetInst::X86MovqToXmm,
                    Writable(select.dst),
                    dst_bits,
                ));
            }
            ty if ty.is_ptr()
                || ty
                    .bit_size()
                    .and_then(|size| size.fixed_bits())
                    .is_some_and(|bits| bits > 32) =>
            {
                self.emit_select_i64_like(ctx, select.dst, cond_i32, select.v1, select.v2, ty);
            }
            ty if ty
                .bit_size()
                .and_then(|size| size.fixed_bits())
                .is_some_and(|bits| bits <= 32) =>
            {
                self.emit_select_i32(ctx, select.dst, cond_i32, select.v1, select.v2);
            }
            _ => {
                panic!("unsupported x86_64 select type: {:?}", dst_ty);
            }
        }

        Ok(SelectResult::Replace)
    }
}

/// x86_64 专属的 Context 扩展实现
pub struct X86SelectionContext<'a> {
    pub vregs: veloc_lir::VRegBuilder<'a>,
    pub features: generated::FeatureSet,
}
impl LoweringContext for X86SelectionContext<'_> {
    fn alloc_tmp(&mut self, like: Reg) -> Reg {
        let data = self
            .vregs
            .get(like.as_vreg().expect("temporary exemplar must be virtual"))
            .clone();
        self.vregs.alloc(data)
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
