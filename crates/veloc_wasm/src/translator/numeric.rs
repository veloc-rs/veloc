use super::WasmTranslator;
use crate::vm::TrapCode;
use veloc::ir::{FloatCC, IntCC, Type as VelocType, Value};
use wasmparser::{BinaryReaderError, Operator};

impl<'a> WasmTranslator<'a> {
    pub(super) fn translate_numeric(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
        match op {
            Operator::I32Add | Operator::I64Add => self.bin(|b, l, r| b.iadd(l, r)),
            Operator::I32Sub | Operator::I64Sub => self.bin(|b, l, r| b.isub(l, r)),
            Operator::I32Mul | Operator::I64Mul => self.bin(|b, l, r| b.imul(l, r)),
            Operator::I32DivS => self.translate_div_s(false),
            Operator::I64DivS => self.translate_div_s(true),
            Operator::I32DivU => self.translate_div_u(false),
            Operator::I64DivU => self.translate_div_u(true),
            Operator::I32RemS => self.translate_rem_s(false),
            Operator::I64RemS => self.translate_rem_s(true),
            Operator::I32RemU => self.translate_rem_u(false),
            Operator::I64RemU => self.translate_rem_u(true),
            Operator::I32And | Operator::I64And => self.bin(|b, l, r| b.iand(l, r)),
            Operator::I32Or | Operator::I64Or => self.bin(|b, l, r| b.ior(l, r)),
            Operator::I32Xor | Operator::I64Xor => self.bin(|b, l, r| b.ixor(l, r)),
            Operator::I32Shl | Operator::I64Shl => self.bin(|b, l, r| b.ishl(l, r)),
            Operator::I32ShrS | Operator::I64ShrS => self.bin(|b, l, r| b.ishr_s(l, r)),
            Operator::I32ShrU | Operator::I64ShrU => self.bin(|b, l, r| b.ishr_u(l, r)),
            Operator::I32Rotl | Operator::I64Rotl => self.bin(|b, l, r| b.irotl(l, r)),
            Operator::I32Rotr | Operator::I64Rotr => self.bin(|b, l, r| b.irotr(l, r)),
            Operator::I32Eq | Operator::I64Eq => self.bin_cmp(|b, l, r| b.eq(l, r)),
            Operator::I32Ne | Operator::I64Ne => self.bin_cmp(|b, l, r| b.ne(l, r)),
            Operator::I32LtS | Operator::I64LtS => self.bin_cmp(|b, l, r| b.lt_s(l, r)),
            Operator::I32LtU | Operator::I64LtU => self.bin_cmp(|b, l, r| b.lt_u(l, r)),
            Operator::I32GtS | Operator::I64GtS => self.bin_cmp(|b, l, r| b.gt_s(l, r)),
            Operator::I32GtU | Operator::I64GtU => self.bin_cmp(|b, l, r| b.gt_u(l, r)),
            Operator::I32LeS | Operator::I64LeS => self.bin_cmp(|b, l, r| b.le_s(l, r)),
            Operator::I32LeU | Operator::I64LeU => self.bin_cmp(|b, l, r| b.le_u(l, r)),
            Operator::I32GeS | Operator::I64GeS => self.bin_cmp(|b, l, r| b.ge_s(l, r)),
            Operator::I32GeU | Operator::I64GeU => self.bin_cmp(|b, l, r| b.ge_u(l, r)),
            Operator::I32Eqz | Operator::I64Eqz => self.un_cmp(|b, v| b.ieqz(v)),
            Operator::I32Clz | Operator::I64Clz => self.un(|b, v| b.iclz(v)),
            Operator::I32Ctz | Operator::I64Ctz => self.un(|b, v| b.ictz(v)),
            Operator::I32Popcnt | Operator::I64Popcnt => self.un(|b, v| b.ipopcnt(v)),

            Operator::I32Extend8S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I8);
                b.extend_s(tmp, VelocType::I32)
            }),
            Operator::I32Extend16S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I16);
                b.extend_s(tmp, VelocType::I32)
            }),
            Operator::I64Extend8S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I8);
                b.extend_s(tmp, VelocType::I64)
            }),
            Operator::I64Extend16S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I16);
                b.extend_s(tmp, VelocType::I64)
            }),
            Operator::I64Extend32S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I32);
                b.extend_s(tmp, VelocType::I64)
            }),
            Operator::I32WrapI64 => self.un(|b, v| b.wrap(v, VelocType::I32)),
            Operator::I64ExtendI32S => self.un(|b, v| b.extend_s(v, VelocType::I64)),
            Operator::I64ExtendI32U => self.un(|b, v| b.extend_u(v, VelocType::I64)),

            Operator::F64Add => self.bin(|b, l, r| b.fadd(l, r)),
            Operator::F64Sub => self.bin(|b, l, r| b.fsub(l, r)),
            Operator::F64Mul => self.bin(|b, l, r| b.fmul(l, r)),
            Operator::F64Div => self.bin(|b, l, r| b.fdiv(l, r)),
            Operator::F32Add => self.bin(|b, l, r| b.fadd(l, r)),
            Operator::F32Sub => self.bin(|b, l, r| b.fsub(l, r)),
            Operator::F32Mul => self.bin(|b, l, r| b.fmul(l, r)),
            Operator::F32Div => self.bin(|b, l, r| b.fdiv(l, r)),
            Operator::F32Neg => self.un(|b, v| b.fneg(v)),
            Operator::F32Abs => self.un(|b, v| b.fabs(v)),
            Operator::F64Neg => self.un(|b, v| b.fneg(v)),
            Operator::F64Abs => self.un(|b, v| b.fabs(v)),
            Operator::F32Sqrt => self.un(|b, v| b.fsqrt(v)),
            Operator::F64Sqrt => self.un(|b, v| b.fsqrt(v)),
            Operator::F32Ceil => self.un(|b, v| b.fceil(v)),
            Operator::F64Ceil => self.un(|b, v| b.fceil(v)),
            Operator::F32Floor => self.un(|b, v| b.ffloor(v)),
            Operator::F64Floor => self.un(|b, v| b.ffloor(v)),
            Operator::F32Trunc => self.un(|b, v| b.ftrunc(v)),
            Operator::F64Trunc => self.un(|b, v| b.ftrunc(v)),
            Operator::F32Nearest => self.un(|b, v| b.fnearest(v)),
            Operator::F64Nearest => self.un(|b, v| b.fnearest(v)),

            Operator::F32Min => self.translate_fmin_fmax(false, true),
            Operator::F64Min => self.translate_fmin_fmax(true, true),
            Operator::F32Max => self.translate_fmin_fmax(false, false),
            Operator::F64Max => self.translate_fmin_fmax(true, false),
            Operator::F32Copysign => self.bin(|b, l, r| b.fcopysign(l, r)),
            Operator::F64Copysign => self.bin(|b, l, r| b.fcopysign(l, r)),

            Operator::F32Eq => self.bin_cmp(|b, l, r| b.feq(l, r)),
            Operator::F32Ne => self.bin_cmp(|b, l, r| b.fne(l, r)),
            Operator::F32Lt => self.bin_cmp(|b, l, r| b.flt(l, r)),
            Operator::F32Gt => self.bin_cmp(|b, l, r| b.fgt(l, r)),
            Operator::F32Le => self.bin_cmp(|b, l, r| b.fle(l, r)),
            Operator::F32Ge => self.bin_cmp(|b, l, r| b.fge(l, r)),
            Operator::F64Eq => self.bin_cmp(|b, l, r| b.feq(l, r)),
            Operator::F64Ne => self.bin_cmp(|b, l, r| b.fne(l, r)),
            Operator::F64Lt => self.bin_cmp(|b, l, r| b.flt(l, r)),
            Operator::F64Gt => self.bin_cmp(|b, l, r| b.fgt(l, r)),
            Operator::F64Le => self.bin_cmp(|b, l, r| b.fle(l, r)),
            Operator::F64Ge => self.bin_cmp(|b, l, r| b.fge(l, r)),

            Operator::F64PromoteF32 => self.un(|b, v| b.float_promote(v, VelocType::F64)),
            Operator::F32DemoteF64 => self.un(|b, v| b.float_demote(v, VelocType::F32)),
            Operator::F32ConvertI32S => self.un(|b, v| b.int_to_float_s(v, VelocType::F32)),
            Operator::F32ConvertI32U => self.un(|b, v| b.int_to_float_u(v, VelocType::F32)),
            Operator::F32ConvertI64S => self.un(|b, v| b.int_to_float_s(v, VelocType::F32)),
            Operator::F32ConvertI64U => self.un(|b, v| b.int_to_float_u(v, VelocType::F32)),
            Operator::I32TruncF32S => self.translate_trunc(true, VelocType::F32, VelocType::I32),
            Operator::I32TruncF32U => self.translate_trunc(false, VelocType::F32, VelocType::I32),
            Operator::F64ConvertI32S => self.un(|b, v| b.int_to_float_s(v, VelocType::F64)),
            Operator::F64ConvertI32U => self.un(|b, v| b.int_to_float_u(v, VelocType::F64)),
            Operator::F64ConvertI64S => self.un(|b, v| b.int_to_float_s(v, VelocType::F64)),
            Operator::F64ConvertI64U => self.un(|b, v| b.int_to_float_u(v, VelocType::F64)),
            Operator::I32TruncF64S => self.translate_trunc(true, VelocType::F64, VelocType::I32),
            Operator::I32TruncF64U => self.translate_trunc(false, VelocType::F64, VelocType::I32),
            Operator::I64TruncF64S => self.translate_trunc(true, VelocType::F64, VelocType::I64),
            Operator::I64TruncF64U => self.translate_trunc(false, VelocType::F64, VelocType::I64),
            Operator::I64TruncF32S => self.translate_trunc(true, VelocType::F32, VelocType::I64),
            Operator::I64TruncF32U => self.translate_trunc(false, VelocType::F32, VelocType::I64),
            Operator::I32TruncSatF32S => self.trunc_sat(true, VelocType::I32),
            Operator::I32TruncSatF32U => self.trunc_sat(false, VelocType::I32),
            Operator::I32TruncSatF64S => self.trunc_sat(true, VelocType::I32),
            Operator::I32TruncSatF64U => self.trunc_sat(false, VelocType::I32),
            Operator::I64TruncSatF32S => self.trunc_sat(true, VelocType::I64),
            Operator::I64TruncSatF32U => self.trunc_sat(false, VelocType::I64),
            Operator::I64TruncSatF64S => self.trunc_sat(true, VelocType::I64),
            Operator::I64TruncSatF64U => self.trunc_sat(false, VelocType::I64),
            Operator::I32ReinterpretF32 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::I32);
                self.stack.push(res);
            }
            Operator::I64ReinterpretF64 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::I64);
                self.stack.push(res);
            }
            Operator::F32ReinterpretI32 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::F32);
                self.stack.push(res);
            }
            Operator::F64ReinterpretI64 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::F64);
                self.stack.push(res);
            }
            _ => unreachable!("Non-numeric operator in translate_numeric"),
        }
        Ok(())
    }

    fn bin<F>(&mut self, f: F)
    where
        F: FnOnce(&mut veloc::ir::InstBuilder, Value, Value) -> Value,
    {
        let r = self.pop();
        let l = self.pop();
        let res = f(&mut self.builder.ins(), l, r);
        self.stack.push(res);
    }

    fn bin_cmp<F>(&mut self, f: F)
    where
        F: FnOnce(&mut veloc::ir::InstBuilder, Value, Value) -> Value,
    {
        let r = self.pop();
        let l = self.pop();
        let cond = f(&mut self.builder.ins(), l, r);
        let true_val = self.builder.ins().iconst(VelocType::I32, 1);
        let false_val = self.builder.ins().iconst(VelocType::I32, 0);
        let res = self.builder.ins().select(cond, true_val, false_val);
        self.stack.push(res);
    }

    fn un<F>(&mut self, f: F)
    where
        F: FnOnce(&mut veloc::ir::InstBuilder, Value) -> Value,
    {
        let v = self.pop();
        let res = f(&mut self.builder.ins(), v);
        self.stack.push(res);
    }

    fn un_cmp<F>(&mut self, f: F)
    where
        F: FnOnce(&mut veloc::ir::InstBuilder, Value) -> Value,
    {
        let v = self.pop();
        let cond = f(&mut self.builder.ins(), v);
        let true_val = self.builder.ins().iconst(VelocType::I32, 1);
        let false_val = self.builder.ins().iconst(VelocType::I32, 0);
        let res = self.builder.ins().select(cond, true_val, false_val);
        self.stack.push(res);
    }

    pub(super) fn translate_trunc(
        &mut self,
        is_signed: bool,
        src_ty: VelocType,
        dst_ty: VelocType,
    ) {
        let val = self.pop();
        let is_nan = self.builder.ins().fne(val, val);
        self.trap_if(is_nan, TrapCode::InvalidConversionToInteger);

        let (is_over, is_under) = match (dst_ty, src_ty, is_signed) {
            (VelocType::I32, VelocType::F32, true) => {
                let upper = self.builder.ins().f32const(2147483648.0);
                let lower = self.builder.ins().f32const(-2147483648.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().flt(val, lower),
                )
            }
            (VelocType::I32, VelocType::F32, false) => {
                let upper = self.builder.ins().f32const(4294967296.0);
                let lower = self.builder.ins().f32const(-1.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().fle(val, lower),
                )
            }
            (VelocType::I32, VelocType::F64, true) => {
                let upper = self.builder.ins().f64const(2147483648.0);
                let lower = self.builder.ins().f64const(-2147483649.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().fle(val, lower),
                )
            }
            (VelocType::I32, VelocType::F64, false) => {
                let upper = self.builder.ins().f64const(4294967296.0);
                let lower = self.builder.ins().f64const(-1.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().fle(val, lower),
                )
            }
            (VelocType::I64, VelocType::F32, true) => {
                let upper = self.builder.ins().f32const(9223372036854775808.0);
                let lower = self.builder.ins().f32const(-9223372036854775808.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().flt(val, lower),
                )
            }
            (VelocType::I64, VelocType::F32, false) => {
                let upper = self.builder.ins().f32const(18446744073709551616.0);
                let lower = self.builder.ins().f32const(-1.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().fle(val, lower),
                )
            }
            (VelocType::I64, VelocType::F64, true) => {
                let upper = self.builder.ins().f64const(9223372036854775808.0);
                let lower = self.builder.ins().f64const(-9223372036854775808.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().flt(val, lower),
                )
            }
            (VelocType::I64, VelocType::F64, false) => {
                let upper = self.builder.ins().f64const(18446744073709551616.0);
                let lower = self.builder.ins().f64const(-1.0);
                (
                    self.builder.ins().fge(val, upper),
                    self.builder.ins().fle(val, lower),
                )
            }
            _ => unreachable!(),
        };

        self.trap_if(is_over, TrapCode::IntegerOverflow);
        self.trap_if(is_under, TrapCode::IntegerOverflow);

        let res = if is_signed {
            self.builder.ins().float_to_int_s(val, dst_ty)
        } else {
            self.builder.ins().float_to_int_u(val, dst_ty)
        };
        self.stack.push(res);
    }

    /// Helper for saturating float-to-int truncation
    #[inline(always)]
    fn trunc_sat(&mut self, is_signed: bool, dst_ty: VelocType) {
        let val = self.pop();
        let res = if is_signed {
            self.builder.ins().float_to_int_sat_s(val, dst_ty)
        } else {
            self.builder.ins().float_to_int_sat_u(val, dst_ty)
        };
        self.stack.push(res);
    }

    pub(super) fn translate_fmin_fmax(&mut self, is_64: bool, is_min: bool) {
        let ty = if is_64 {
            VelocType::F64
        } else {
            VelocType::F32
        };
        let r = self.pop();
        let l = self.pop();
        let zero = self.builder.ins().fconst(ty, 0);
        let l_is_zero = self.builder.ins().fcmp(FloatCC::Eq, l, zero);
        let r_is_zero = self.builder.ins().fcmp(FloatCC::Eq, r, zero);
        let both_zero = self.builder.ins().iand(l_is_zero, r_is_zero);
        let res_var = self.new_var(ty);

        self.builder.if_else(
            both_zero,
            |b| {
                let ity = if is_64 {
                    VelocType::I64
                } else {
                    VelocType::I32
                };
                let l_bits = b.ins().reinterpret(l, ity);
                let r_bits = b.ins().reinterpret(r, ity);
                let res_bits = if is_min {
                    b.ins().ior(l_bits, r_bits)
                } else {
                    b.ins().iand(l_bits, r_bits)
                };
                let res = b.ins().reinterpret(res_bits, ty);
                b.def_var(res_var, res);
            },
            |b| {
                let res = if is_min {
                    b.ins().fmin(l, r)
                } else {
                    b.ins().fmax(l, r)
                };
                b.def_var(res_var, res);
            },
        );

        self.terminated = false;
        let final_res = self.builder.use_var(res_var);
        self.stack.push(final_res);
    }

    pub(super) fn translate_div_s(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();
        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);
        let neg_one = self.builder.ins().iconst(ty, -1);
        let min_val = if is_64 { i64::MIN } else { i32::MIN as i64 };
        let min = self.builder.ins().iconst(ty, min_val);
        let is_min = self.builder.ins().icmp(IntCC::Eq, l, min);
        let is_neg_one = self.builder.ins().icmp(IntCC::Eq, r, neg_one);
        let is_overflow = self.builder.ins().iand(is_min, is_neg_one);
        self.trap_if(is_overflow, TrapCode::IntegerOverflow);
        let res = self.builder.ins().idiv_s(l, r);
        self.stack.push(res);
    }

    pub(super) fn translate_div_u(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();
        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);
        let res = self.builder.ins().idiv_u(l, r);
        self.stack.push(res);
    }

    pub(super) fn translate_rem_s(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();
        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);
        let neg_one = self.builder.ins().iconst(ty, -1);
        let min_val = if is_64 { i64::MIN } else { i32::MIN as i64 };
        let min = self.builder.ins().iconst(ty, min_val);
        let is_min = self.builder.ins().icmp(IntCC::Eq, l, min);
        let is_neg_one = self.builder.ins().icmp(IntCC::Eq, r, neg_one);
        let is_overflow = self.builder.ins().iand(is_min, is_neg_one);
        let res_var = self.new_var(ty);

        self.builder.if_else(
            is_overflow,
            |b| {
                let zero_res = b.ins().iconst(ty, 0);
                b.def_var(res_var, zero_res);
            },
            |b| {
                let rem_res = b.ins().irem_s(l, r);
                b.def_var(res_var, rem_res);
            },
        );

        let final_res = self.builder.use_var(res_var);
        self.stack.push(final_res);
    }

    pub(super) fn translate_rem_u(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();
        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);
        let res = self.builder.ins().irem_u(l, r);
        self.stack.push(res);
    }
}
