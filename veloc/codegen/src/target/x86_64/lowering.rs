//! x86_64 Target Lowering
//!
//! 使用 ISLE (Instruction Selection Lowering Expressions) 生成的代码
//! 进行指令选择。

mod frame;
mod legalize;
mod numeric;
mod operand;
mod pass_config;
mod select;

pub use crate::isel::SelectResult;
use crate::passes::lowering::{LegalizeAction, LegalizeResult};
use crate::target::arch::{
    CallConv as TargetCallConv, CpuDescription, LoweringContext, OperandConstraintSet,
    SelectionContext, TargetArch, TargetFrameLowering, TargetInstructionSelector, TargetLegalizer,
    TargetOperandLowering, TargetPassConfig, TargetPostIsel,
};
use crate::target::x86_64::isle::{TargetInst, generated};
use alloc::vec::Vec;
pub use frame::X86_64FrameLowering;
pub use legalize::X86_64Legalizer;
pub use operand::X86_64OperandLowering;
pub use pass_config::{X86_64PassConfig, X86_64PostIsel};
pub use select::X86_64Selector;
use veloc_lir::InstBuild;
use veloc_lir::RegisterBank;
use veloc_lir::{
    GenericOpcode, InstExtra, InstField, InstId, MachineFunction, MachineOpcode, Reg, VReg,
    Writable,
};
use veloc_mir::{FloatCC, IntCC, Type, TypeInfo};

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
    pub cpu: CpuDescription,
}

impl X86_64Lowering {
    pub fn new(cpu: CpuDescription) -> Self {
        Self { cpu }
    }

    fn alloc_gpr_temp(&self, mfunc: &mut MachineFunction, ty: Type) -> Reg {
        mfunc.editor().alloc_vreg_in_bank(ty, RegisterBank::GPR)
    }

    fn emit_legalize_constant_reg(
        mfunc: &mut MachineFunction,
        output: &mut Vec<InstId>,
        ty: Type,
        imm: i64,
    ) -> Reg {
        let reg = mfunc.editor().alloc_vreg(ty);
        output.push(mfunc.editor().writer().constant(Writable(reg), imm));
        reg
    }

    fn emit_legalize_binary_reg(
        mfunc: &mut MachineFunction,
        output: &mut Vec<InstId>,
        opcode: GenericOpcode,
        ty: Type,
        lhs: Reg,
        rhs: Reg,
    ) -> Reg {
        let dst = mfunc.editor().alloc_vreg(ty);
        output.push(mfunc.editor().writer().binary(
            MachineOpcode::Generic(opcode),
            Writable(dst),
            lhs,
            rhs,
        ));
        dst
    }

    fn legalize_ctpop_into(
        mfunc: &mut MachineFunction,
        output: &mut Vec<InstId>,
        src: Reg,
        dst: Reg,
        ty: Type,
    ) -> Result<(), crate::error::Error> {
        let is_i32 = ty == Type::I32;
        let is_i64 = ty == Type::I64;
        if !is_i32 && !is_i64 {
            panic!(
                "unsupported ctpop type during x86_64 legalization: {:?}",
                ty
            );
        }

        let shift1 = Self::emit_legalize_constant_reg(mfunc, output, ty, 1);
        let shift2 = Self::emit_legalize_constant_reg(mfunc, output, ty, 2);
        let shift4 = Self::emit_legalize_constant_reg(mfunc, output, ty, 4);
        let shift8 = Self::emit_legalize_constant_reg(mfunc, output, ty, 8);
        let shift16 = Self::emit_legalize_constant_reg(mfunc, output, ty, 16);

        let mask1 = Self::emit_legalize_constant_reg(
            mfunc,
            output,
            ty,
            if is_i32 {
                0x5555_5555
            } else {
                0x5555_5555_5555_5555u64 as i64
            },
        );
        let mask2 = Self::emit_legalize_constant_reg(
            mfunc,
            output,
            ty,
            if is_i32 {
                0x3333_3333
            } else {
                0x3333_3333_3333_3333u64 as i64
            },
        );
        let mask3 = Self::emit_legalize_constant_reg(
            mfunc,
            output,
            ty,
            if is_i32 {
                0x0f0f_0f0f
            } else {
                0x0f0f_0f0f_0f0f_0f0fu64 as i64
            },
        );
        let final_mask =
            Self::emit_legalize_constant_reg(mfunc, output, ty, if is_i32 { 0x3f } else { 0x7f });

        let x1 =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, src, shift1);
        let x2 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::And, ty, x1, mask1);
        let x3 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Sub, ty, src, x2);
        let x4 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::And, ty, x3, mask2);
        let x5 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x3, shift2);
        let x6 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::And, ty, x5, mask2);
        let x7 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Add, ty, x4, x6);
        let x8 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x7, shift4);
        let x9 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Add, ty, x7, x8);
        let x10 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::And, ty, x9, mask3);
        let x11 =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x10, shift8);
        let x12 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Add, ty, x10, x11);
        let x13 =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x12, shift16);
        let x14 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Add, ty, x12, x13);

        let reduced = if is_i64 {
            let shift32 = Self::emit_legalize_constant_reg(mfunc, output, ty, 32);
            let x15 = Self::emit_legalize_binary_reg(
                mfunc,
                output,
                GenericOpcode::Lshr,
                ty,
                x14,
                shift32,
            );
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Add, ty, x14, x15)
        } else {
            x14
        };

        let pop = Self::emit_legalize_binary_reg(
            mfunc,
            output,
            GenericOpcode::And,
            ty,
            reduced,
            final_mask,
        );
        if pop != dst {
            output.push(mfunc.editor().writer().copy(Writable(dst), pop));
        }
        Ok(())
    }

    fn legalize_cttz_into(
        mfunc: &mut MachineFunction,
        output: &mut Vec<InstId>,
        src: Reg,
        dst: Reg,
        ty: Type,
    ) -> Result<(), crate::error::Error> {
        let bits = if ty == Type::I32 {
            32
        } else if ty == Type::I64 {
            64
        } else {
            panic!("unsupported cttz type during x86_64 legalization: {:?}", ty);
        };

        let zero = Self::emit_legalize_constant_reg(mfunc, output, ty, 0);
        let one = Self::emit_legalize_constant_reg(mfunc, output, ty, 1);
        let bit_width = Self::emit_legalize_constant_reg(mfunc, output, ty, bits);
        let is_zero = mfunc.editor().alloc_vreg(Type::BOOL);
        output.push(
            mfunc
                .editor()
                .writer()
                .icmp(Writable(is_zero), src, zero, IntCC::Eq),
        );

        let neg = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Sub, ty, zero, src);
        let lowbit =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::And, ty, src, neg);
        let lowbit_minus_one =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Sub, ty, lowbit, one);
        let pop = mfunc.editor().alloc_vreg(ty);
        Self::legalize_ctpop_into(mfunc, output, lowbit_minus_one, pop, ty)?;

        output.push(
            mfunc
                .editor()
                .writer()
                .select(Writable(dst), is_zero, bit_width, pop),
        );
        Ok(())
    }

    fn legalize_ctlz_into(
        mfunc: &mut MachineFunction,
        output: &mut Vec<InstId>,
        src: Reg,
        dst: Reg,
        ty: Type,
    ) -> Result<(), crate::error::Error> {
        let bits = if ty == Type::I32 {
            32
        } else if ty == Type::I64 {
            64
        } else {
            panic!("unsupported ctlz type during x86_64 legalization: {:?}", ty);
        };

        let zero = Self::emit_legalize_constant_reg(mfunc, output, ty, 0);
        let bit_width = Self::emit_legalize_constant_reg(mfunc, output, ty, bits);
        let is_zero = mfunc.editor().alloc_vreg(Type::BOOL);
        output.push(
            mfunc
                .editor()
                .writer()
                .icmp(Writable(is_zero), src, zero, IntCC::Eq),
        );

        let shift1 = Self::emit_legalize_constant_reg(mfunc, output, ty, 1);
        let shift2 = Self::emit_legalize_constant_reg(mfunc, output, ty, 2);
        let shift4 = Self::emit_legalize_constant_reg(mfunc, output, ty, 4);
        let shift8 = Self::emit_legalize_constant_reg(mfunc, output, ty, 8);
        let shift16 = Self::emit_legalize_constant_reg(mfunc, output, ty, 16);

        let x1 =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, src, shift1);
        let x2 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Or, ty, src, x1);
        let x3 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x2, shift2);
        let x4 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Or, ty, x2, x3);
        let x5 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x4, shift4);
        let x6 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Or, ty, x4, x5);
        let x7 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x6, shift8);
        let x8 = Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Or, ty, x6, x7);
        let x9 =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Lshr, ty, x8, shift16);
        let mut filled =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Or, ty, x8, x9);

        if ty == Type::I64 {
            let shift32 = Self::emit_legalize_constant_reg(mfunc, output, ty, 32);
            let x10 = Self::emit_legalize_binary_reg(
                mfunc,
                output,
                GenericOpcode::Lshr,
                ty,
                filled,
                shift32,
            );
            filled =
                Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Or, ty, filled, x10);
        }

        let pop = mfunc.editor().alloc_vreg(ty);
        Self::legalize_ctpop_into(mfunc, output, filled, pop, ty)?;
        let clz =
            Self::emit_legalize_binary_reg(mfunc, output, GenericOpcode::Sub, ty, bit_width, pop);
        output.push(
            mfunc
                .editor()
                .writer()
                .select(Writable(dst), is_zero, bit_width, clz),
        );
        Ok(())
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
    pub cpu: CpuDescription,
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
        self.cpu.has_feature("BMI2")
    }
    fn has_avx2(&self) -> bool {
        self.cpu.has_feature("AVX2")
    }
}

impl crate::target::arch::TargetFeatures for X86SelectionContext<'_> {
    fn has_feature(&self, feature: &str) -> bool {
        self.cpu.has_feature(feature)
    }
}
