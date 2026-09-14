//! Unsigned conversions on baseline SSE2, without target-specific runtime calls.
use super::*;
use veloc_lir::InstBuild;

fn unary(
    mfunc: &mut MachineFunction<LegalizedLir>,
    out: &mut Vec<InstId>,
    opcode: GenericOpcode,
    ty: Type,
    src: Reg,
) -> Reg {
    let dst = mfunc.alloc_vreg(ty);
    out.push(
        mfunc
            .writer()
            .unary(MachineOpcode::Generic(opcode), Writable(dst), src),
    );
    dst
}

impl X86_64Lowering {
    pub(super) fn unsigned_conversion(
        &self,
        mfunc: &mut MachineFunction<LegalizedLir>,
        out: &mut Vec<InstId>,
        opcode: GenericOpcode,
        dst: Reg,
        src: Reg,
    ) {
        let dst_ty = mfunc.vreg_data(dst).ty;
        let src_ty = mfunc.vreg_data(src).ty;
        let result = if opcode == GenericOpcode::G_UITOFP {
            if src_ty == Type::I32 {
                let extended = unary(mfunc, out, GenericOpcode::G_ZEXT, Type::I64, src);
                unary(mfunc, out, GenericOpcode::G_SITOFP, dst_ty, extended)
            } else {
                // Preserve the low bit as a sticky bit before rounding, avoiding
                // double rounding when an unsigned value exceeds i64::MAX.
                let one = self.emit_legalize_constant_reg(mfunc, out, Type::I64, 1);
                let zero = self.emit_legalize_constant_reg(mfunc, out, Type::I64, 0);
                let half = self.emit_legalize_binary_reg(
                    mfunc,
                    out,
                    GenericOpcode::G_LSHR,
                    Type::I64,
                    src,
                    one,
                );
                let low = self.emit_legalize_binary_reg(
                    mfunc,
                    out,
                    GenericOpcode::G_AND,
                    Type::I64,
                    src,
                    one,
                );
                let rounded = self.emit_legalize_binary_reg(
                    mfunc,
                    out,
                    GenericOpcode::G_OR,
                    Type::I64,
                    half,
                    low,
                );
                let half_float = unary(mfunc, out, GenericOpcode::G_SITOFP, dst_ty, rounded);
                let doubled = self.emit_legalize_binary_reg(
                    mfunc,
                    out,
                    GenericOpcode::G_FADD,
                    dst_ty,
                    half_float,
                    half_float,
                );
                let direct = unary(mfunc, out, GenericOpcode::G_SITOFP, dst_ty, src);
                let high = mfunc.alloc_vreg(Type::BOOL);
                out.push(mfunc.writer().icmp(Writable(high), src, zero, IntCC::LtS));
                let result = mfunc.alloc_vreg(dst_ty);
                out.push(
                    mfunc
                        .writer()
                        .select(Writable(result), high, doubled, direct),
                );
                result
            }
        } else {
            assert_eq!(opcode, GenericOpcode::G_FPTOUI);
            if dst_ty == Type::I32 {
                let wide = unary(mfunc, out, GenericOpcode::G_FPTOSI, Type::I64, src);
                unary(mfunc, out, GenericOpcode::G_TRUNC, Type::I32, wide)
            } else {
                // For the upper half of u64, subtract the exact power of two,
                // convert the remainder as signed, then restore the high bit.
                let (bits_ty, bits) = if src_ty == Type::F32 {
                    (Type::I32, (9223372036854775808.0f32).to_bits() as i64)
                } else {
                    (Type::I64, (9223372036854775808.0f64).to_bits() as i64)
                };
                let bits = self.emit_legalize_constant_reg(mfunc, out, bits_ty, bits);
                let threshold = unary(mfunc, out, GenericOpcode::G_BITCAST, src_ty, bits);
                let high = mfunc.alloc_vreg(Type::BOOL);
                out.push(
                    mfunc
                        .writer()
                        .fcmp(Writable(high), src, threshold, FloatCC::Ge),
                );
                let reduced = self.emit_legalize_binary_reg(
                    mfunc,
                    out,
                    GenericOpcode::G_FSUB,
                    src_ty,
                    src,
                    threshold,
                );
                let converted = unary(mfunc, out, GenericOpcode::G_FPTOSI, Type::I64, reduced);
                let sign = self.emit_legalize_constant_reg(mfunc, out, Type::I64, i64::MIN);
                let restored = self.emit_legalize_binary_reg(
                    mfunc,
                    out,
                    GenericOpcode::G_XOR,
                    Type::I64,
                    converted,
                    sign,
                );
                let direct = unary(mfunc, out, GenericOpcode::G_FPTOSI, Type::I64, src);
                let result = mfunc.alloc_vreg(dst_ty);
                out.push(
                    mfunc
                        .writer()
                        .select(Writable(result), high, restored, direct),
                );
                result
            }
        };
        out.push(mfunc.writer().copy(Writable(dst), result));
    }
}
