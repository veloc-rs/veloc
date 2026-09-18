//! Unsigned conversions on baseline SSE2, without target-specific runtime calls.
use super::*;
use veloc_lir::{InstBuild, InstRead};

fn unary(
    mfunc: &mut MachineFunction,
    out: &mut Vec<InstId>,
    opcode: GenericOpcode,
    ty: Type,
    src: Reg,
) -> Reg {
    let dst = mfunc.editor().alloc_vreg(ty);
    out.push(
        mfunc
            .editor()
            .writer()
            .write(MachineOpcode::Generic(opcode), &[dst], &[src], &[]),
    );
    dst
}

pub(super) fn unsigned_conversion(
    inst_id: InstId,
    mfunc: &mut MachineFunction,
) -> crate::error::Result<LegalizeResult> {
    let inst = mfunc.inst(inst_id);
    let opcode = inst.generic_opcode().unwrap();
    let veloc_lir::InstView::UnaryReg(unary_inst) = inst.view() else {
        unreachable!()
    };
    let (dst, src) = (unary_inst.dst, unary_inst.src);
    let mut output = Vec::new();
    let out = &mut output;
    let dst_ty = mfunc.vreg_data(dst).ty;
    let src_ty = mfunc.vreg_data(src).ty;
    let result = if opcode == GenericOpcode::Uitofp {
        if src_ty == Type::I32 {
            let extended = unary(mfunc, out, GenericOpcode::Zext, Type::I64, src);
            unary(mfunc, out, GenericOpcode::Sitofp, dst_ty, extended)
        } else {
            // Preserve the low bit as a sticky bit before rounding, avoiding
            // double rounding when an unsigned value exceeds i64::MAX.
            let one = constant(mfunc, out, Type::I64, 1);
            let zero = constant(mfunc, out, Type::I64, 0);
            let half = binary(mfunc, out, GenericOpcode::Lshr, Type::I64, src, one);
            let low = binary(mfunc, out, GenericOpcode::And, Type::I64, src, one);
            let rounded = binary(mfunc, out, GenericOpcode::Or, Type::I64, half, low);
            let half_float = unary(mfunc, out, GenericOpcode::Sitofp, dst_ty, rounded);
            let doubled = binary(
                mfunc,
                out,
                GenericOpcode::Fadd,
                dst_ty,
                half_float,
                half_float,
            );
            let direct = unary(mfunc, out, GenericOpcode::Sitofp, dst_ty, src);
            let high = mfunc.editor().alloc_vreg(Type::BOOL);
            out.push(
                mfunc
                    .editor()
                    .writer()
                    .icmp(Writable(high), src, zero, IntCC::LtS),
            );
            let result = mfunc.editor().alloc_vreg(dst_ty);
            out.push(
                mfunc
                    .editor()
                    .writer()
                    .select(Writable(result), high, doubled, direct),
            );
            result
        }
    } else {
        assert_eq!(opcode, GenericOpcode::Fptoui);
        if dst_ty == Type::I32 {
            let wide = unary(mfunc, out, GenericOpcode::Fptosi, Type::I64, src);
            unary(mfunc, out, GenericOpcode::Trunc, Type::I32, wide)
        } else {
            // For the upper half of u64, subtract the exact power of two,
            // convert the remainder as signed, then restore the high bit.
            let (bits_ty, bits) = if src_ty == Type::F32 {
                (Type::I32, (9223372036854775808.0f32).to_bits() as i64)
            } else {
                (Type::I64, (9223372036854775808.0f64).to_bits() as i64)
            };
            let bits = constant(mfunc, out, bits_ty, bits);
            let threshold = unary(mfunc, out, GenericOpcode::Bitcast, src_ty, bits);
            let high = mfunc.editor().alloc_vreg(Type::BOOL);
            out.push(
                mfunc
                    .editor()
                    .writer()
                    .fcmp(Writable(high), src, threshold, FloatCC::Ge),
            );
            let reduced = binary(mfunc, out, GenericOpcode::Fsub, src_ty, src, threshold);
            let converted = unary(mfunc, out, GenericOpcode::Fptosi, Type::I64, reduced);
            let sign = constant(mfunc, out, Type::I64, i64::MIN);
            let restored = binary(mfunc, out, GenericOpcode::Xor, Type::I64, converted, sign);
            let direct = unary(mfunc, out, GenericOpcode::Fptosi, Type::I64, src);
            let result = mfunc.editor().alloc_vreg(dst_ty);
            out.push(
                mfunc
                    .editor()
                    .writer()
                    .select(Writable(result), high, restored, direct),
            );
            result
        }
    };
    out.push(mfunc.editor().writer().copy(Writable(dst), result));
    Ok(LegalizeResult::Replace(output))
}

fn constant(mfunc: &mut MachineFunction, output: &mut Vec<InstId>, ty: Type, imm: i64) -> Reg {
    let reg = mfunc.editor().alloc_vreg(ty);
    output.push(mfunc.editor().writer().constant(Writable(reg), imm));
    reg
}

fn binary(
    mfunc: &mut MachineFunction,
    output: &mut Vec<InstId>,
    opcode: GenericOpcode,
    ty: Type,
    lhs: Reg,
    rhs: Reg,
) -> Reg {
    let dst = mfunc.editor().alloc_vreg(ty);
    output.push(mfunc.editor().writer().write(
        MachineOpcode::Generic(opcode),
        &[dst],
        &[lhs, rhs],
        &[],
    ));
    dst
}
