//! x86_64 Target Lowering
//!
//! 使用 ISLE (Instruction Selection Lowering Expressions) 生成的代码
//! 进行指令选择。

mod frame;
mod legalize;
mod operand;
mod pass_config;
mod regbank;
mod select;

pub use crate::isel::SelectResult;
use crate::mir::{
    GenericOpcode, InstExtra, InstId, MachineFunction, MachineInst, MachineOpcode, MachineOperand,
    Reg, VReg, Writable,
};
use crate::passes::lowering::{LegalizeAction, LegalizeResult};
use crate::pipeline::stages::{LegalizedMir, PreIselPrepared, RegAllocated, SelectedMir};
use crate::regalloc::regbank_select::RegisterBank;
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
pub use regbank::X86_64RegBankSelect;
pub use select::X86_64Selector;
use smallvec::smallvec;
use veloc_ir::{FloatCC, IntCC, Type};

/// x86_64 专属的 Context 扩展 (架构私有)
pub trait X86LoweringContext: LoweringContext {
    fn has_bmi2(&self) -> bool;

    fn has_avx2(&self) -> bool;
}

fn x86_mov_opcode_for_type(ty: Type) -> Result<TargetInst, crate::error::Error> {
    if ty == Type::F32 {
        Ok(TargetInst::X86Movss)
    } else if ty == Type::F64 {
        Ok(TargetInst::X86Movsd)
    } else if ty.min_size_bytes().is_some_and(|bytes| bytes <= 4) {
        Ok(TargetInst::X86Mov32)
    } else if ty.min_size_bytes().is_some_and(|bytes| bytes <= 8) || ty.is_ptr() {
        Ok(TargetInst::X86Mov64)
    } else {
        panic!("unsupported type for x86_64 move: {:?}", ty);
    }
}

fn x86_copy_type_for_regs<S>(
    mfunc: &MachineFunction<S>,
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

fn build_x86_copy_inst<S>(
    mfunc: &MachineFunction<S>,
    dst: Reg,
    src: Reg,
) -> Result<MachineInst, crate::error::Error> {
    let ty = x86_copy_type_for_regs(mfunc, dst, src)?;
    let opcode = x86_mov_opcode_for_type(ty)?;
    Ok(MachineInst::build_tied_binary(
        MachineOpcode::Target(opcode.as_u32()),
        Writable(dst),
        src,
    ))
}

fn build_target_inst(
    opcode: TargetInst,
    operands: smallvec::SmallVec<[MachineOperand; 4]>,
) -> MachineInst {
    MachineInst::build_generic(MachineOpcode::Target(opcode.as_u32()), operands)
}

fn build_target_imm(opcode: TargetInst, dst: Writable<Reg>, imm: i64) -> MachineInst {
    build_target_inst(
        opcode,
        smallvec![MachineOperand::Def(dst), MachineOperand::Imm(imm)],
    )
}

fn build_target_unary(opcode: TargetInst, dst: Writable<Reg>, src: Reg) -> MachineInst {
    MachineInst::build_unary(MachineOpcode::Target(opcode.as_u32()), dst, src)
}

fn build_target_binary_uses(opcode: TargetInst, lhs: Reg, rhs: Reg) -> MachineInst {
    build_target_inst(
        opcode,
        smallvec![MachineOperand::Use(lhs), MachineOperand::Use(rhs)],
    )
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

    fn alloc_gpr_temp<S>(&self, mfunc: &mut MachineFunction<S>, ty: Type) -> Reg {
        mfunc.alloc_vreg_in_bank(ty, RegisterBank::GPR)
    }

    fn push_legalized_inst(
        &self,
        mfunc: &mut MachineFunction<LegalizedMir>,
        output: &mut Vec<InstId>,
        inst: MachineInst,
    ) -> InstId {
        let inst_id = mfunc.alloc_inst(inst);
        output.push(inst_id);
        inst_id
    }

    fn emit_legalize_constant_reg(
        &self,
        mfunc: &mut MachineFunction<LegalizedMir>,
        output: &mut Vec<InstId>,
        ty: Type,
        imm: i64,
    ) -> Reg {
        let reg = mfunc.alloc_vreg(ty);
        self.push_legalized_inst(
            mfunc,
            output,
            MachineInst::build_constant(Writable(reg), imm),
        );
        reg
    }

    fn emit_legalize_binary_reg(
        &self,
        mfunc: &mut MachineFunction<LegalizedMir>,
        output: &mut Vec<InstId>,
        opcode: GenericOpcode,
        ty: Type,
        lhs: Reg,
        rhs: Reg,
    ) -> Reg {
        let dst = mfunc.alloc_vreg(ty);
        self.push_legalized_inst(
            mfunc,
            output,
            MachineInst::build_binary(MachineOpcode::Generic(opcode), Writable(dst), lhs, rhs),
        );
        dst
    }

    fn legalize_ctpop_into(
        &self,
        mfunc: &mut MachineFunction<LegalizedMir>,
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

        let shift1 = self.emit_legalize_constant_reg(mfunc, output, ty, 1);
        let shift2 = self.emit_legalize_constant_reg(mfunc, output, ty, 2);
        let shift4 = self.emit_legalize_constant_reg(mfunc, output, ty, 4);
        let shift8 = self.emit_legalize_constant_reg(mfunc, output, ty, 8);
        let shift16 = self.emit_legalize_constant_reg(mfunc, output, ty, 16);

        let mask1 = self.emit_legalize_constant_reg(
            mfunc,
            output,
            ty,
            if is_i32 {
                0x5555_5555
            } else {
                0x5555_5555_5555_5555u64 as i64
            },
        );
        let mask2 = self.emit_legalize_constant_reg(
            mfunc,
            output,
            ty,
            if is_i32 {
                0x3333_3333
            } else {
                0x3333_3333_3333_3333u64 as i64
            },
        );
        let mask3 = self.emit_legalize_constant_reg(
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
            self.emit_legalize_constant_reg(mfunc, output, ty, if is_i32 { 0x3f } else { 0x7f });

        let x1 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, src, shift1);
        let x2 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_AND, ty, x1, mask1);
        let x3 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_SUB, ty, src, x2);
        let x4 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_AND, ty, x3, mask2);
        let x5 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x3, shift2);
        let x6 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_AND, ty, x5, mask2);
        let x7 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_ADD, ty, x4, x6);
        let x8 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x7, shift4);
        let x9 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_ADD, ty, x7, x8);
        let x10 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_AND, ty, x9, mask3);
        let x11 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x10, shift8);
        let x12 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_ADD, ty, x10, x11);
        let x13 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x12, shift16);
        let x14 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_ADD, ty, x12, x13);

        let reduced = if is_i64 {
            let shift32 = self.emit_legalize_constant_reg(mfunc, output, ty, 32);
            let x15 = self.emit_legalize_binary_reg(
                mfunc,
                output,
                GenericOpcode::G_LSHR,
                ty,
                x14,
                shift32,
            );
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_ADD, ty, x14, x15)
        } else {
            x14
        };

        let pop = self.emit_legalize_binary_reg(
            mfunc,
            output,
            GenericOpcode::G_AND,
            ty,
            reduced,
            final_mask,
        );
        if pop != dst {
            self.push_legalized_inst(mfunc, output, MachineInst::build_copy(Writable(dst), pop));
        }
        Ok(())
    }

    fn legalize_cttz_into(
        &self,
        mfunc: &mut MachineFunction<LegalizedMir>,
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

        let zero = self.emit_legalize_constant_reg(mfunc, output, ty, 0);
        let one = self.emit_legalize_constant_reg(mfunc, output, ty, 1);
        let bit_width = self.emit_legalize_constant_reg(mfunc, output, ty, bits);
        let is_zero = mfunc.alloc_vreg(Type::BOOL);
        self.push_legalized_inst(
            mfunc,
            output,
            MachineInst::build_icmp(Writable(is_zero), src, zero, IntCC::Eq),
        );

        let neg = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_SUB, ty, zero, src);
        let lowbit =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_AND, ty, src, neg);
        let lowbit_minus_one =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_SUB, ty, lowbit, one);
        let pop = mfunc.alloc_vreg(ty);
        self.legalize_ctpop_into(mfunc, output, lowbit_minus_one, pop, ty)?;

        self.push_legalized_inst(
            mfunc,
            output,
            MachineInst::build_select(Writable(dst), is_zero, bit_width, pop),
        );
        Ok(())
    }

    fn legalize_ctlz_into(
        &self,
        mfunc: &mut MachineFunction<LegalizedMir>,
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

        let zero = self.emit_legalize_constant_reg(mfunc, output, ty, 0);
        let bit_width = self.emit_legalize_constant_reg(mfunc, output, ty, bits);
        let is_zero = mfunc.alloc_vreg(Type::BOOL);
        self.push_legalized_inst(
            mfunc,
            output,
            MachineInst::build_icmp(Writable(is_zero), src, zero, IntCC::Eq),
        );

        let shift1 = self.emit_legalize_constant_reg(mfunc, output, ty, 1);
        let shift2 = self.emit_legalize_constant_reg(mfunc, output, ty, 2);
        let shift4 = self.emit_legalize_constant_reg(mfunc, output, ty, 4);
        let shift8 = self.emit_legalize_constant_reg(mfunc, output, ty, 8);
        let shift16 = self.emit_legalize_constant_reg(mfunc, output, ty, 16);

        let x1 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, src, shift1);
        let x2 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_OR, ty, src, x1);
        let x3 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x2, shift2);
        let x4 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_OR, ty, x2, x3);
        let x5 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x4, shift4);
        let x6 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_OR, ty, x4, x5);
        let x7 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x6, shift8);
        let x8 = self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_OR, ty, x6, x7);
        let x9 =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_LSHR, ty, x8, shift16);
        let mut filled =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_OR, ty, x8, x9);

        if ty == Type::I64 {
            let shift32 = self.emit_legalize_constant_reg(mfunc, output, ty, 32);
            let x10 = self.emit_legalize_binary_reg(
                mfunc,
                output,
                GenericOpcode::G_LSHR,
                ty,
                filled,
                shift32,
            );
            filled =
                self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_OR, ty, filled, x10);
        }

        let pop = mfunc.alloc_vreg(ty);
        self.legalize_ctpop_into(mfunc, output, filled, pop, ty)?;
        let clz =
            self.emit_legalize_binary_reg(mfunc, output, GenericOpcode::G_SUB, ty, bit_width, pop);
        self.push_legalized_inst(
            mfunc,
            output,
            MachineInst::build_select(Writable(dst), is_zero, bit_width, clz),
        );
        Ok(())
    }

    fn normalize_cond_to_i32(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
        cond: Reg,
        cond_ty: Type,
    ) -> Reg {
        let test_opcode = if cond_ty.min_size_bytes().is_some_and(|bytes| bytes <= 4) {
            TargetInst::X86Test32
        } else {
            TargetInst::X86Test64
        };
        let cond_byte = self.alloc_gpr_temp(ctx.mfunc, Type::I8);
        let cond_i32 = self.alloc_gpr_temp(ctx.mfunc, Type::I32);

        ctx.selected
            .push(build_target_binary_uses(test_opcode, cond, cond));
        ctx.selected.push(build_target_inst(
            TargetInst::X86Setne,
            smallvec![MachineOperand::Def(Writable(cond_byte))],
        ));
        ctx.selected.push(build_target_unary(
            TargetInst::X86Movzx8to32,
            Writable(cond_i32),
            cond_byte,
        ));

        cond_i32
    }

    fn emit_select_i32(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
        dst: Reg,
        cond_i32: Reg,
        true_val: Reg,
        false_val: Reg,
    ) {
        let mask = self.alloc_gpr_temp(ctx.mfunc, Type::I32);
        let diff = self.alloc_gpr_temp(ctx.mfunc, Type::I32);

        ctx.selected
            .push(build_target_imm(TargetInst::X86Mov32Imm, Writable(mask), 0));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Sub32.as_u32()),
            Writable(mask),
            cond_i32,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
            Writable(diff),
            true_val,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Xor32.as_u32()),
            Writable(diff),
            false_val,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
            Writable(dst),
            false_val,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86And32.as_u32()),
            Writable(diff),
            mask,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Xor32.as_u32()),
            Writable(dst),
            diff,
        ));
    }

    fn emit_select_i64_like(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
        dst: Reg,
        cond_i32: Reg,
        true_val: Reg,
        false_val: Reg,
        wide_ty: Type,
    ) {
        let cond_i64 = self.alloc_gpr_temp(ctx.mfunc, Type::I64);
        let mask = self.alloc_gpr_temp(ctx.mfunc, Type::I64);
        let diff = self.alloc_gpr_temp(ctx.mfunc, wide_ty);

        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
            Writable(cond_i64),
            cond_i32,
        ));
        ctx.selected.push(build_target_imm(
            TargetInst::X86Mov64Imm32,
            Writable(mask),
            0,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Sub64.as_u32()),
            Writable(mask),
            cond_i64,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
            Writable(diff),
            true_val,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Xor64.as_u32()),
            Writable(diff),
            false_val,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
            Writable(dst),
            false_val,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86And64.as_u32()),
            Writable(diff),
            mask,
        ));
        ctx.selected.push(MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86Xor64.as_u32()),
            Writable(dst),
            diff,
        ));
    }

    fn select_fcmp(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
        inst: &MachineInst,
    ) -> Result<SelectResult, crate::error::Error> {
        let fcmp = inst.as_fcmp().unwrap_or_else(|err| {
            panic!("invalid fcmp instruction during x86_64 selection: {}", err);
        });
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

        ctx.selected
            .push(build_target_binary_uses(compare_opcode, fcmp.lhs, fcmp.rhs));

        let emit_setcc_i32 =
            |ctx: &mut SelectionContext<'_, PreIselPrepared>, opcode: TargetInst| -> Reg {
                let tmp8 = self.alloc_gpr_temp(ctx.mfunc, Type::I8);
                let tmp32 = self.alloc_gpr_temp(ctx.mfunc, Type::I32);
                ctx.selected.push(build_target_inst(
                    opcode,
                    smallvec![MachineOperand::Def(Writable(tmp8))],
                ));
                ctx.selected.push(build_target_unary(
                    TargetInst::X86Movzx8to32,
                    Writable(tmp32),
                    tmp8,
                ));
                tmp32
            };

        match fcmp.cc {
            FloatCC::Eq => {
                let is_eq = emit_setcc_i32(ctx, TargetInst::X86Sete);
                let ordered = emit_setcc_i32(ctx, TargetInst::X86Setnp);
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
                    Writable(fcmp.dst),
                    is_eq,
                ));
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86And32.as_u32()),
                    Writable(fcmp.dst),
                    ordered,
                ));
            }
            FloatCC::Ne => {
                let is_ne = emit_setcc_i32(ctx, TargetInst::X86Setne);
                let unordered = emit_setcc_i32(ctx, TargetInst::X86Setp);
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
                    Writable(fcmp.dst),
                    is_ne,
                ));
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Or32.as_u32()),
                    Writable(fcmp.dst),
                    unordered,
                ));
            }
            other => {
                panic!(
                    "x86_64 special FCMP selector should only handle Eq/Ne, found {:?}",
                    other
                );
            }
        }

        Ok(SelectResult::Replace)
    }

    fn select_select(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
        inst: &MachineInst,
    ) -> Result<SelectResult, crate::error::Error> {
        let select = inst.as_select().unwrap_or_else(|err| {
            panic!(
                "invalid select instruction during x86_64 selection: {}",
                err
            );
        });
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
                    TargetInst::X86MovdFromXmm,
                    Writable(true_bits),
                    select.v1,
                ));
                ctx.selected.push(build_target_unary(
                    TargetInst::X86MovdFromXmm,
                    Writable(false_bits),
                    select.v2,
                ));
                self.emit_select_i32(ctx, dst_bits, cond_i32, true_bits, false_bits);
                ctx.selected.push(build_target_unary(
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
                    TargetInst::X86MovqFromXmm,
                    Writable(true_bits),
                    select.v1,
                ));
                ctx.selected.push(build_target_unary(
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
                    TargetInst::X86MovqToXmm,
                    Writable(select.dst),
                    dst_bits,
                ));
            }
            ty if ty.is_ptr() || ty.min_size_bytes().is_some_and(|bytes| bytes > 4) => {
                self.emit_select_i64_like(ctx, select.dst, cond_i32, select.v1, select.v2, ty);
            }
            ty if ty.min_size_bytes().is_some_and(|bytes| bytes <= 4) => {
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
pub struct X86SelectionContext<'a, 'b> {
    pub base: &'a mut SelectionContext<'b, PreIselPrepared>,
    pub cpu: CpuDescription,
}

impl<'a, 'b> LoweringContext for X86SelectionContext<'a, 'b> {
    fn get_type(&self, vreg: VReg) -> Type {
        self.base.get_type(vreg)
    }

    fn get_bank(&self, vreg: VReg) -> Option<RegisterBank> {
        self.base.get_bank(vreg)
    }

    fn get_vreg(&self, inst: &MachineInst, index: usize) -> Option<VReg> {
        self.base.get_vreg(inst, index)
    }
}

impl<'a, 'b> X86LoweringContext for X86SelectionContext<'a, 'b> {
    fn has_bmi2(&self) -> bool {
        self.cpu.has_feature("BMI2")
    }

    fn has_avx2(&self) -> bool {
        self.cpu.has_feature("AVX2")
    }
}
