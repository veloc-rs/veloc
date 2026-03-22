//! x86_64 Target Lowering
//!
//! 使用 ISLE (Instruction Selection Lowering Expressions) 生成的代码
//! 进行指令选择。

pub use crate::isel::SelectResult;
use crate::mir::{
    CallCallee, GenericOpcode, InstExtra, InstId, MachineFunction, MachineInst, MachineOpcode,
    MachineOperand, Reg, VReg, Writable,
};
use crate::regalloc::regbank_select::RegisterBank;
use crate::target::arch::{
    CallConv as TargetCallConv, CpuDescription, LegalizerInfo, LoweringContext,
    OperandConstraintSet, OperandConstraintStage, SelectionContext, TargetArch, TargetLowering,
};
use crate::target::x86_64::isle::{TargetInst, generated};
use alloc::vec;
use alloc::vec::Vec;
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
    } else if ty.size_bytes() <= 4 {
        Ok(TargetInst::X86Mov32)
    } else if ty.size_bytes() <= 8 {
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
    mfunc: &MachineFunction,
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

fn build_target_tied_binary(opcode: TargetInst, dst: Writable<Reg>, src: Reg) -> MachineInst {
    MachineInst::build_tied_binary(MachineOpcode::Target(opcode.as_u32()), dst, src)
}

fn setcc_opcode_for_intcc(cc: IntCC) -> TargetInst {
    match cc {
        IntCC::Eq => TargetInst::X86Sete,
        IntCC::Ne => TargetInst::X86Setne,
        IntCC::LtS => TargetInst::X86Setl,
        IntCC::LtU => TargetInst::X86Setb,
        IntCC::GtS => TargetInst::X86Setg,
        IntCC::GtU => TargetInst::X86Seta,
        IntCC::LeS => TargetInst::X86Setle,
        IntCC::LeU => TargetInst::X86Setbe,
        IntCC::GeS => TargetInst::X86Setge,
        IntCC::GeU => TargetInst::X86Setae,
    }
}

/// x86_64 Lowering
pub struct X86_64Lowering {
    legalizer_info: LegalizerInfo,
    /// 当前 target instance 选中的 CPU 描述。
    pub cpu: CpuDescription,
}

impl X86_64Lowering {
    pub fn new(cpu: CpuDescription) -> Self {
        let legalizer_info = Self::create_legalizer_info();
        Self {
            legalizer_info,
            cpu,
        }
    }

    /// 创建合法化信息
    fn create_legalizer_info() -> LegalizerInfo {
        use crate::mir::GenericOpcode;

        let mut info = LegalizerInfo::new();

        // x86_64 支持 I32 和 I64 类型
        for opcode in [
            GenericOpcode::G_ADD,
            GenericOpcode::G_SUB,
            GenericOpcode::G_MUL,
            GenericOpcode::G_AND,
            GenericOpcode::G_OR,
            GenericOpcode::G_XOR,
        ] {
            info.get_action_definitions_builder(opcode)
                .legal_for_types(&[Type::I32, Type::I64])
                .widen_scalar_for_type(Type::I8)
                .widen_scalar_for_type(Type::I16);
        }

        // 移位操作
        for opcode in [
            GenericOpcode::G_SHL,
            GenericOpcode::G_LSHR,
            GenericOpcode::G_ASHR,
        ] {
            info.get_action_definitions_builder(opcode)
                .legal_for_types(&[Type::I32, Type::I64]);
        }

        // 其他操作
        for opcode in [
            GenericOpcode::G_ICMP,
            GenericOpcode::G_LOAD,
            GenericOpcode::G_STORE,
            GenericOpcode::G_OFFSET_LOAD,
            GenericOpcode::G_OFFSET_STORE,
            GenericOpcode::G_INDEXED_LOAD,
            GenericOpcode::G_INDEXED_STORE,
            GenericOpcode::G_CONSTANT,
        ] {
            info.get_action_definitions_builder(opcode)
                .legal_for_types(&[Type::I32, Type::I64, Type::F32, Type::F64]);
        }

        for opcode in [
            GenericOpcode::G_COPY,
            GenericOpcode::G_BITCAST,
            GenericOpcode::G_FCONSTANT,
            GenericOpcode::G_FADD,
            GenericOpcode::G_FSUB,
            GenericOpcode::G_FMUL,
            GenericOpcode::G_FDIV,
        ] {
            info.get_action_definitions_builder(opcode)
                .legal_for_types(&[Type::F32, Type::F64]);
        }

        info.get_action_definitions_builder(GenericOpcode::G_BRJT)
            .lower_for(vec![Type::I32]);
        info.get_action_definitions_builder(GenericOpcode::G_CTPOP)
            .lower_for(vec![Type::I32, Type::I32])
            .lower_for(vec![Type::I64, Type::I64]);
        info.get_action_definitions_builder(GenericOpcode::G_CTLZ)
            .lower_for(vec![Type::I32, Type::I32])
            .lower_for(vec![Type::I64, Type::I64]);
        info.get_action_definitions_builder(GenericOpcode::G_CTTZ)
            .lower_for(vec![Type::I32, Type::I32])
            .lower_for(vec![Type::I64, Type::I64]);

        info
    }

    fn push_legalized_inst(
        &self,
        mfunc: &mut MachineFunction,
        output: &mut Vec<InstId>,
        inst: MachineInst,
    ) -> InstId {
        let inst_id = mfunc.alloc_inst(inst);
        output.push(inst_id);
        inst_id
    }

    fn emit_legalize_constant_reg(
        &self,
        mfunc: &mut MachineFunction,
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
        mfunc: &mut MachineFunction,
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

    fn normalize_cond_to_i32(&self, ctx: &mut SelectionContext, cond: Reg, cond_ty: Type) -> Reg {
        let test_opcode = if cond_ty.size_bytes() <= 4 {
            TargetInst::X86Test32
        } else {
            TargetInst::X86Test64
        };
        let cond_byte = ctx.mfunc.alloc_vreg(Type::I8);
        let cond_i32 = ctx.mfunc.alloc_vreg(Type::I32);

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
        ctx: &mut SelectionContext,
        dst: Reg,
        cond_i32: Reg,
        true_val: Reg,
        false_val: Reg,
    ) {
        let mask = ctx.mfunc.alloc_vreg(Type::I32);
        let diff = ctx.mfunc.alloc_vreg(Type::I32);

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
        ctx: &mut SelectionContext,
        dst: Reg,
        cond_i32: Reg,
        true_val: Reg,
        false_val: Reg,
        wide_ty: Type,
    ) {
        let cond_i64 = ctx.mfunc.alloc_vreg(Type::I64);
        let mask = ctx.mfunc.alloc_vreg(Type::I64);
        let diff = ctx.mfunc.alloc_vreg(wide_ty);

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

    fn select_icmp(
        &self,
        ctx: &mut SelectionContext,
        inst: &MachineInst,
    ) -> Result<SelectResult, crate::error::Error> {
        let icmp = inst.as_icmp().unwrap_or_else(|err| {
            panic!("invalid icmp instruction during x86_64 selection: {}", err);
        });
        let rhs = icmp.rhs.unwrap_or_else(|| {
            panic!("x86_64 icmp selection expects a rhs operand");
        });
        let cc = icmp.cc.unwrap_or_else(|| {
            panic!("x86_64 icmp selection expects an integer condition code");
        });
        let lhs_ty = if icmp.lhs.is_vreg() {
            ctx.mfunc.vreg_data(icmp.lhs).ty
        } else {
            Type::I64
        };
        let cmp_opcode = if lhs_ty.size_bytes() <= 4 {
            TargetInst::X86Cmp32
        } else {
            TargetInst::X86Cmp64
        };
        let cond_byte = ctx.mfunc.alloc_vreg(Type::I8);

        ctx.selected
            .push(build_target_binary_uses(cmp_opcode, icmp.lhs, rhs));
        ctx.selected.push(build_target_inst(
            setcc_opcode_for_intcc(cc),
            smallvec![MachineOperand::Def(Writable(cond_byte))],
        ));
        ctx.selected.push(build_target_unary(
            TargetInst::X86Movzx8to32,
            Writable(icmp.dst),
            cond_byte,
        ));

        Ok(SelectResult::Replace)
    }

    fn select_ieqz(
        &self,
        ctx: &mut SelectionContext,
        inst: &MachineInst,
    ) -> Result<SelectResult, crate::error::Error> {
        let unary = inst.as_unary_reg()?;
        let ty = ctx.mfunc.vreg_data(unary.src).ty;
        let test_opcode = if ty.size_bytes() <= 4 {
            TargetInst::X86Test32
        } else {
            TargetInst::X86Test64
        };
        let cond_byte = ctx.mfunc.alloc_vreg(Type::I8);

        ctx.selected
            .push(build_target_binary_uses(test_opcode, unary.src, unary.src));
        ctx.selected.push(build_target_inst(
            TargetInst::X86Sete,
            smallvec![MachineOperand::Def(Writable(cond_byte))],
        ));
        ctx.selected.push(build_target_unary(
            TargetInst::X86Movzx8to32,
            Writable(unary.dst),
            cond_byte,
        ));

        Ok(SelectResult::Replace)
    }

    fn select_fcmp(
        &self,
        ctx: &mut SelectionContext,
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

        let emit_setcc_i32 = |ctx: &mut SelectionContext, opcode: TargetInst| -> Reg {
            let tmp8 = ctx.mfunc.alloc_vreg(Type::I8);
            let tmp32 = ctx.mfunc.alloc_vreg(Type::I32);
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
            FloatCC::Lt => {
                let tmp = emit_setcc_i32(ctx, TargetInst::X86Setb);
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
                    Writable(fcmp.dst),
                    tmp,
                ));
            }
            FloatCC::Gt => {
                let tmp = emit_setcc_i32(ctx, TargetInst::X86Seta);
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
                    Writable(fcmp.dst),
                    tmp,
                ));
            }
            FloatCC::Le => {
                let tmp = emit_setcc_i32(ctx, TargetInst::X86Setbe);
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
                    Writable(fcmp.dst),
                    tmp,
                ));
            }
            FloatCC::Ge => {
                let tmp = emit_setcc_i32(ctx, TargetInst::X86Setae);
                ctx.selected.push(MachineInst::build_tied_binary(
                    MachineOpcode::Target(TargetInst::X86Mov32.as_u32()),
                    Writable(fcmp.dst),
                    tmp,
                ));
            }
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
        }

        Ok(SelectResult::Replace)
    }

    fn select_select(
        &self,
        ctx: &mut SelectionContext,
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
                let true_bits = ctx.mfunc.alloc_vreg(Type::I32);
                let false_bits = ctx.mfunc.alloc_vreg(Type::I32);
                let dst_bits = ctx.mfunc.alloc_vreg(Type::I32);

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
                let true_bits = ctx.mfunc.alloc_vreg(Type::I64);
                let false_bits = ctx.mfunc.alloc_vreg(Type::I64);
                let dst_bits = ctx.mfunc.alloc_vreg(Type::I64);

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
            ty if ty.is_ptr() || ty.size_bytes() > 4 => {
                self.emit_select_i64_like(ctx, select.dst, cond_i32, select.v1, select.v2, ty);
            }
            ty if ty.size_bytes() <= 4 => {
                self.emit_select_i32(ctx, select.dst, cond_i32, select.v1, select.v2);
            }
            _ => {
                panic!("unsupported x86_64 select type: {:?}", dst_ty);
            }
        }

        Ok(SelectResult::Replace)
    }

    fn select_tied_int_binary(
        &self,
        ctx: &mut SelectionContext,
        inst: &MachineInst,
        opcode: GenericOpcode,
    ) -> Result<SelectResult, crate::error::Error> {
        let binary = inst.as_binary_reg().unwrap_or_else(|err| {
            panic!(
                "invalid tied integer binary instruction during x86_64 selection: {}",
                err
            );
        });
        if binary.dst != binary.lhs {
            panic!(
                "expected tied lhs after operand-constraint preparation, got dst={:?}, lhs={:?}",
                binary.dst, binary.lhs
            );
        }

        let dst_ty = if binary.dst.is_vreg() {
            ctx.mfunc.vreg_data(binary.dst).ty
        } else {
            Type::I64
        };

        let target_opcode = match (opcode, dst_ty) {
            (GenericOpcode::G_ADD, Type::I32) => TargetInst::X86Add32,
            (GenericOpcode::G_SUB, Type::I32) => TargetInst::X86Sub32,
            (GenericOpcode::G_MUL, Type::I32) => TargetInst::X86IMul32,
            (GenericOpcode::G_AND, Type::I32) => TargetInst::X86And32,
            (GenericOpcode::G_OR, Type::I32) => TargetInst::X86Or32,
            (GenericOpcode::G_XOR, Type::I32) => TargetInst::X86Xor32,
            (GenericOpcode::G_ADD, Type::I64 | Type::PTR) => TargetInst::X86Add64,
            (GenericOpcode::G_SUB, Type::I64) => TargetInst::X86Sub64,
            (GenericOpcode::G_MUL, Type::I64) => TargetInst::X86IMul64,
            (GenericOpcode::G_AND, Type::I64) => TargetInst::X86And64,
            (GenericOpcode::G_OR, Type::I64) => TargetInst::X86Or64,
            (GenericOpcode::G_XOR, Type::I64) => TargetInst::X86Xor64,
            (GenericOpcode::G_PTR_ADD, Type::I64 | Type::PTR) => TargetInst::X86Add64,
            _ => {
                panic!(
                    "unsupported x86 integer binary selection for opcode {:?} with type {:?}",
                    opcode, dst_ty
                );
            }
        };

        ctx.selected.push(build_target_tied_binary(
            target_opcode,
            Writable(binary.dst),
            binary.rhs,
        ));
        Ok(SelectResult::InPlace)
    }

    fn select_tied_shift(
        &self,
        ctx: &mut SelectionContext,
        inst: &MachineInst,
        opcode: GenericOpcode,
    ) -> Result<SelectResult, crate::error::Error> {
        let binary = inst.as_binary_reg().unwrap_or_else(|err| {
            panic!(
                "invalid tied shift instruction during x86_64 selection: {}",
                err
            );
        });
        if binary.dst != binary.lhs {
            panic!(
                "expected tied lhs after operand-constraint preparation, got dst={:?}, lhs={:?}",
                binary.dst, binary.lhs
            );
        }

        let dst_ty = if binary.dst.is_vreg() {
            ctx.mfunc.vreg_data(binary.dst).ty
        } else {
            Type::I64
        };

        let target_opcode = match (opcode, dst_ty) {
            (GenericOpcode::G_SHL, Type::I32) => TargetInst::X86Shl32Cl,
            (GenericOpcode::G_LSHR, Type::I32) => TargetInst::X86Shr32Cl,
            (GenericOpcode::G_ASHR, Type::I32) => TargetInst::X86Sar32Cl,
            (GenericOpcode::G_SHL, Type::I64) => TargetInst::X86Shl64Cl,
            (GenericOpcode::G_LSHR, Type::I64) => TargetInst::X86Shr64Cl,
            (GenericOpcode::G_ASHR, Type::I64) => TargetInst::X86Sar64Cl,
            _ => {
                panic!(
                    "unsupported x86 shift selection for opcode {:?} with type {:?}",
                    opcode, dst_ty
                );
            }
        };

        ctx.selected.push(build_target_tied_binary(
            target_opcode,
            Writable(binary.dst),
            binary.rhs,
        ));
        Ok(SelectResult::InPlace)
    }
}

/// x86_64 专属的 Context 扩展实现
pub struct X86SelectionContext<'a, 'b> {
    pub base: &'a mut SelectionContext<'b>,
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

impl TargetLowering for X86_64Lowering {
    fn finalize_stack_frame(&self, mfunc: &mut MachineFunction, call_conv: TargetCallConv) {
        let preserved_regs = call_conv.preserved_regs(TargetArch::X86_64);
        let mut used_callee_saved = Vec::new();
        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
                for reg in mfunc.dfg[inst_id].defs().chain(mfunc.dfg[inst_id].uses()) {
                    if preserved_regs.contains(&reg) && !used_callee_saved.contains(&reg) {
                        used_callee_saved.push(reg);
                    }
                }
            }
        }

        mfunc.stack_frame.callee_saved_size = (used_callee_saved.len() as u32) * 8;
        mfunc.stack_frame.used_callee_saved = used_callee_saved;

        let mut total = mfunc.stack_frame.local_size
            + mfunc.stack_frame.callee_saved_size
            + mfunc.stack_frame.arg_size;
        let align = 16;
        let misalign = total % align;
        if misalign != 0 {
            total += align - misalign;
        }
        mfunc.stack_frame.total_size = total;
    }

    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction) {
        use crate::mir::MachineOpcode;
        use crate::target::x86_64::isle::{REG_RBP, REG_RSP, TargetInst};

        let stack_size = mfunc.stack_frame.total_size;
        let saved_regs = mfunc.stack_frame.used_callee_saved.clone();
        let local_size = mfunc.stack_frame.local_size as i32;

        // 1. 在入口块插入序言指令
        if !mfunc.blocks.is_empty() {
            let mut pending_prologue = Vec::new();

            // push rbp
            let push_inst = MachineInst::build_generic(
                MachineOpcode::Target(TargetInst::X86PushRbp.as_u32()),
                smallvec::SmallVec::new(),
            );
            pending_prologue.push(mfunc.alloc_inst(push_inst));

            // mov rbp, rsp
            let mov_inst = MachineInst::build_generic(
                MachineOpcode::Target(TargetInst::X86MovRbpRsp.as_u32()),
                smallvec::SmallVec::new(),
            );
            pending_prologue.push(mfunc.alloc_inst(mov_inst));

            if stack_size > 0 {
                // sub rsp, stack_size
                let sub_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Sub64ri.as_u32()),
                    smallvec::smallvec![
                        crate::mir::MachineOperand::TiedDefUse(crate::mir::Writable(REG_RSP)),
                        crate::mir::MachineOperand::Imm(stack_size as i64),
                    ],
                );
                pending_prologue.push(mfunc.alloc_inst(sub_inst));
            }

            for (idx, reg) in saved_regs.iter().copied().enumerate() {
                let offset = -(local_size + ((idx as i32 + 1) * 8));
                let save_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Store64.as_u32()),
                    smallvec::smallvec![
                        crate::mir::MachineOperand::Use(reg),
                        crate::mir::MachineOperand::Use(REG_RBP),
                        crate::mir::MachineOperand::Imm(offset as i64),
                    ],
                );
                pending_prologue.push(mfunc.alloc_inst(save_inst));
            }

            mfunc
                .rewrite_block(0, |cursor| {
                    if !pending_prologue.is_empty() {
                        for inst_id in pending_prologue.drain(..) {
                            cursor.emit_existing_before(inst_id);
                        }
                    }
                    cursor.emit_existing_before(cursor.current_inst_id());
                    cursor.remove_current();
                    Ok::<(), crate::error::Error>(())
                })
                .expect("x86_64 prologue rewriting should not fail");
        }

        // 2. 在每个返回指令前插入尾声指令
        for block_idx in 0..mfunc.blocks.len() {
            mfunc
                .rewrite_block(block_idx, |cursor| {
                    let inst_id = cursor.current_inst_id();
                    let is_ret = matches!(
                        cursor.current_inst().opcode,
                        MachineOpcode::Target(code) if code == TargetInst::X86Ret.as_u32()
                    );

                    if is_ret {
                        for (idx, reg) in saved_regs.iter().copied().enumerate().rev() {
                            let offset = -(local_size + ((idx as i32 + 1) * 8));
                            let restore_inst = MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86Load64.as_u32()),
                                smallvec::smallvec![
                                    crate::mir::MachineOperand::Def(crate::mir::Writable(reg)),
                                    crate::mir::MachineOperand::Use(REG_RBP),
                                    crate::mir::MachineOperand::Imm(offset as i64),
                                ],
                            );
                            cursor.emit_before(restore_inst);
                        }

                        if stack_size > 0 {
                            // add rsp, stack_size
                            let add_inst = MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86Add64ri.as_u32()),
                                smallvec::smallvec![
                                    crate::mir::MachineOperand::TiedDefUse(crate::mir::Writable(
                                        REG_RSP
                                    )),
                                    crate::mir::MachineOperand::Imm(stack_size as i64),
                                ],
                            );
                            cursor.emit_before(add_inst);
                        }

                        // pop rbp
                        let pop_inst = MachineInst::build_generic(
                            MachineOpcode::Target(TargetInst::X86PopRbp.as_u32()),
                            smallvec::SmallVec::new(),
                        );
                        cursor.emit_before(pop_inst);
                    }

                    cursor.emit_existing_before(inst_id);
                    cursor.remove_current();
                    Ok::<(), crate::error::Error>(())
                })
                .expect("x86_64 prologue/epilogue rewriting should not fail");
        }
    }

    fn legalize_instruction(
        &self,
        inst_id: crate::mir::InstId,
        mfunc: &mut crate::mir::MachineFunction,
        output: &mut Vec<crate::mir::InstId>,
    ) {
        let opcode = mfunc.dfg[inst_id].generic_opcode();
        if let Some(opcode) = opcode {
            match opcode {
                GenericOpcode::G_CTPOP | GenericOpcode::G_CTLZ | GenericOpcode::G_CTTZ => {
                    let inst = mfunc.dfg[inst_id].clone();
                    let unary = inst.as_unary_reg().unwrap_or_else(|err| {
                        panic!(
                            "invalid unary opcode {:?} during x86_64 legalization: {}",
                            inst.opcode, err
                        );
                    });
                    let ty = if unary.dst.is_vreg() {
                        mfunc.vreg_data(unary.dst).ty
                    } else {
                        panic!(
                            "x86_64 legalization expected virtual register destination for {:?}",
                            inst.opcode
                        );
                    };
                    match opcode {
                        GenericOpcode::G_CTPOP => {
                            let _ =
                                self.legalize_ctpop_into(mfunc, output, unary.src, unary.dst, ty);
                        }
                        GenericOpcode::G_CTLZ => {
                            let _ =
                                self.legalize_ctlz_into(mfunc, output, unary.src, unary.dst, ty);
                        }
                        GenericOpcode::G_CTTZ => {
                            let _ =
                                self.legalize_cttz_into(mfunc, output, unary.src, unary.dst, ty);
                        }
                        _ => unreachable!(),
                    };
                    return;
                }
                _ => {}
            }
        }

        if matches!(
            mfunc.dfg[inst_id].opcode,
            MachineOpcode::Generic(crate::mir::GenericOpcode::G_BRJT)
        ) {
            let Some(InstExtra::BrTable(info)) = mfunc.inst_extra(inst_id).cloned() else {
                panic!("missing br_table extra during x86_64 br_table legalization");
            };
            let Ok(brjt) = mfunc.dfg[inst_id].as_branch_table() else {
                panic!("invalid br_table instruction during x86_64 legalization");
            };

            if info.targets.is_empty() {
                return;
            }

            let index = brjt.index;
            let default_target = info.targets.last().unwrap();
            debug_assert!(
                info.targets.iter().all(|target| target.args.is_empty()),
                "edge arguments should be lowered before x86_64 br_table legalization"
            );

            for (case_idx, target) in info.targets[..info.targets.len() - 1].iter().enumerate() {
                let cmp_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Cmp32ri.as_u32()),
                    smallvec::smallvec![
                        MachineOperand::Use(index),
                        MachineOperand::Imm(case_idx as i64),
                    ],
                );
                output.push(mfunc.alloc_inst(cmp_inst));

                let je_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Je.as_u32()),
                    smallvec::smallvec![MachineOperand::Block(target.block)],
                );
                output.push(mfunc.alloc_inst(je_inst));
            }

            let jmp_inst = MachineInst::build_generic(
                MachineOpcode::Target(TargetInst::X86Jmp.as_u32()),
                smallvec::smallvec![MachineOperand::Block(default_target.block)],
            );
            output.push(mfunc.alloc_inst(jmp_inst));
            return;
        }

        // 对于 x86_64，大多数指令可以直接选择
        output.push(inst_id);
    }

    fn select_instruction(
        &self,
        ctx: &mut SelectionContext,
    ) -> Result<SelectResult, crate::error::Error> {
        let cpu = self.cpu;
        let inst = ctx.mfunc.dfg[ctx.inst_id].clone();

        if matches!(inst.opcode, MachineOpcode::Target(_)) {
            return Ok(SelectResult::Keep);
        }

        if let MachineOpcode::Generic(_opcode) = inst.opcode {
            match _opcode {
                GenericOpcode::G_CALL => {
                    let call = ctx.mfunc.as_call(ctx.inst_id);
                    return Ok(match call.shape.callee {
                        CallCallee::Direct(callee) => {
                            ctx.selected.push(MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86Call.as_u32()),
                                smallvec![MachineOperand::Global(callee)],
                            ));
                            SelectResult::Replace
                        }
                        CallCallee::Indirect(callee_reg) => {
                            ctx.selected.push(MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86CallReg.as_u32()),
                                smallvec![MachineOperand::Use(callee_reg)],
                            ));
                            SelectResult::Replace
                        }
                    });
                }
                GenericOpcode::G_CALLIND => {
                    let call = ctx.mfunc.as_call(ctx.inst_id);
                    return Ok(match call.shape.callee {
                        CallCallee::Direct(callee) => {
                            ctx.selected.push(MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86Call.as_u32()),
                                smallvec![MachineOperand::Global(callee)],
                            ));
                            SelectResult::Replace
                        }
                        CallCallee::Indirect(callee_reg) => {
                            ctx.selected.push(MachineInst::build_generic(
                                MachineOpcode::Target(TargetInst::X86CallReg.as_u32()),
                                smallvec![MachineOperand::Use(callee_reg)],
                            ));
                            SelectResult::Replace
                        }
                    });
                }
                GenericOpcode::G_FCMP => return self.select_fcmp(ctx, &inst),
                GenericOpcode::G_SELECT => {
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
                    if dst_ty.is_float() {
                        return self.select_select(ctx, &inst);
                    }
                }
                _ => {}
            }
        }

        let result = {
            let selected = core::mem::take(ctx.selected);
            let mut out = selected;
            let x86_ctx = X86SelectionContext { base: ctx, cpu };
            let res = generated::select_instructions(&x86_ctx, &inst, &mut out);
            *ctx.selected = out;
            res.unwrap_or_else(|err| panic!("x86_64 generated selector failed: {}", err))
        };

        Ok(result)
    }

    fn operand_constraints(
        &self,
        stage: OperandConstraintStage,
        inst: &MachineInst,
        _mfunc: &MachineFunction,
    ) -> OperandConstraintSet {
        match stage {
            OperandConstraintStage::PreSelect => {
                let Some(opcode) = inst.generic_opcode() else {
                    return OperandConstraintSet::default();
                };
                generated::generic_inst_metadata(opcode).operand_constraints()
            }
            OperandConstraintStage::PostSelect => {
                let MachineOpcode::Target(opcode) = inst.opcode else {
                    return OperandConstraintSet::default();
                };
                generated::target_inst_metadata(TargetInst::from_u32(opcode)).operand_constraints()
            }
        }
    }

    fn build_reg_copy(
        &self,
        mfunc: &MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> Result<MachineInst, crate::error::Error> {
        Ok(build_x86_copy_inst(mfunc, dst, src).unwrap_or_else(|err| {
            panic!(
                "failed to build x86_64 reg copy for {:?} <- {:?}: {}",
                dst, src, err
            )
        }))
    }

    fn legalizer_info(&self) -> &LegalizerInfo {
        &self.legalizer_info
    }
}

impl crate::regalloc::regbank_select::TargetRegBankSelect for X86_64Lowering {
    fn suggest_bank(
        &self,
        opcode: crate::mir::GenericOpcode,
        _index: usize,
        ty: Type,
    ) -> Option<crate::regalloc::regbank_select::RegisterBank> {
        use crate::mir::GenericOpcode;
        use crate::regalloc::regbank_select::RegisterBank;

        match opcode {
            // 算术指令通常建议 GPR
            GenericOpcode::G_ADD | GenericOpcode::G_SUB | GenericOpcode::G_MUL => {
                if ty.is_float() {
                    Some(RegisterBank::FPR)
                } else {
                    Some(RegisterBank::GPR)
                }
            }
            // 浮点运算强制建议 FPR
            GenericOpcode::G_FADD
            | GenericOpcode::G_FSUB
            | GenericOpcode::G_FMUL
            | GenericOpcode::G_FDIV => Some(RegisterBank::FPR),
            // 内存操作
            GenericOpcode::G_LOAD
            | GenericOpcode::G_STORE
            | GenericOpcode::G_OFFSET_LOAD
            | GenericOpcode::G_OFFSET_STORE => {
                if ty.is_float() || ty.is_vector() {
                    Some(RegisterBank::FPR)
                } else {
                    Some(RegisterBank::GPR)
                }
            }
            _ => None,
        }
    }
}
