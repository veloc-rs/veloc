use super::super::*;

define_register_handlers! {
    // === I32 Arithmetic ===
    I32Add { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_add(b)));
    }
    I32AddImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a.wrapping_add(imm as i32)));
    }
    I32Sub { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_sub(b)));
    }
    I32SubImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a.wrapping_sub(imm as i32)));
    }
    I32Mul { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_mul(b)));
    }
    I32DivS { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_div(b)));
    }
    I32DivU { dst, src1, src2 } => {
        let (a, b) = (
            get!(src1).unwrap_i32() as u32,
            get!(src2).unwrap_i32() as u32,
        );
        set!(dst, InterpreterValue::i32(a.wrapping_div(b) as i32));
    }
    I32RemS { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_rem(b)));
    }
    I32RemU { dst, src1, src2 } => {
        let (a, b) = (
            get!(src1).unwrap_i32() as u32,
            get!(src2).unwrap_i32() as u32,
        );
        set!(dst, InterpreterValue::i32(a.wrapping_rem(b) as i32));
    }
    I32And { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a & b));
    }
    I32AndImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a & imm as i32));
    }
    I32Or { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a | b));
    }
    I32OrImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a | imm as i32));
    }
    I32Xor { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a ^ b));
    }
    I32XorImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a ^ imm as i32));
    }
    I32Shl { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_shl(b as u32)));
    }
    I32ShlImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a.wrapping_shl(imm as u32)));
    }
    I32ShrS { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.wrapping_shr(b as u32)));
    }
    I32ShrSImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32();
        set!(dst, InterpreterValue::i32(a.wrapping_shr(imm as u32)));
    }
    I32ShrU { dst, src1, src2 } => {
        let (a, b) = (
            get!(src1).unwrap_i32() as u32,
            get!(src2).unwrap_i32() as u32,
        );
        set!(dst, InterpreterValue::i32(a.wrapping_shr(b) as i32));
    }
    I32ShrUImm { dst, src1, imm } => {
        let a = get!(src1).unwrap_i32() as u32;
        set!(
            dst,
            InterpreterValue::i32(a.wrapping_shr(imm as u32) as i32)
        );
    }
    I32RotL { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.rotate_left(b as u32)));
    }
    I32RotR { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
        set!(dst, InterpreterValue::i32(a.rotate_right(b as u32)));
    }
    I32Clz { dst, src } => {
        set!(
            dst,
            InterpreterValue::i32(get!(src).unwrap_i32().leading_zeros() as i32)
        )
    }
    I32Ctz { dst, src } => {
        set!(
            dst,
            InterpreterValue::i32(get!(src).unwrap_i32().trailing_zeros() as i32)
        )
    }
    I32Popcnt { dst, src } => {
        set!(
            dst,
            InterpreterValue::i32(get!(src).unwrap_i32().count_ones() as i32)
        )
    }
    I32Eqz { dst, src_val } => {
        set!(dst, InterpreterValue::bool(get!(src_val).unwrap_i32() == 0))
    }
    I32Eq { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i32() == get!(src2).unwrap_i32())
        )
    }
    I32Ne { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i32() != get!(src2).unwrap_i32())
        )
    }
    I32LtS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i32() < get!(src2).unwrap_i32())
        )
    }
    I32LtU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i32() as u32) < (get!(src2).unwrap_i32() as u32)
            )
        )
    }
    I32LeS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i32() <= get!(src2).unwrap_i32())
        )
    }
    I32LeU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i32() as u32) <= (get!(src2).unwrap_i32() as u32)
            )
        )
    }
    I32GtS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i32() > get!(src2).unwrap_i32())
        )
    }
    I32GtU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i32() as u32) > (get!(src2).unwrap_i32() as u32)
            )
        )
    }
    I32GeS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i32() >= get!(src2).unwrap_i32())
        )
    }
    I32GeU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i32() as u32) >= (get!(src2).unwrap_i32() as u32)
            )
        )
    }

    // === I64 Operations ===
    I64Add { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_add(b)));
    }
    I64AddImm { dst, src1, imm64 } => {
        let a = get!(src1).unwrap_i64();
        set!(dst, InterpreterValue::i64(a.wrapping_add(imm64 as i64)));
    }
    I64Sub { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_sub(b)));
    }
    I64SubImm { dst, src1, imm64 } => {
        let a = get!(src1).unwrap_i64();
        set!(dst, InterpreterValue::i64(a.wrapping_sub(imm64 as i64)));
    }
    I64Mul { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_mul(b)));
    }
    I64DivS { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_div(b)));
    }
    I64DivU { dst, src1, src2 } => {
        let (a, b) = (
            get!(src1).unwrap_i64() as u64,
            get!(src2).unwrap_i64() as u64,
        );
        set!(dst, InterpreterValue::i64(a.wrapping_div(b) as i64));
    }
    I64RemS { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_rem(b)));
    }
    I64RemU { dst, src1, src2 } => {
        let (a, b) = (
            get!(src1).unwrap_i64() as u64,
            get!(src2).unwrap_i64() as u64,
        );
        set!(dst, InterpreterValue::i64(a.wrapping_rem(b) as i64));
    }
    I64And { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a & b));
    }
    I64AndImm { dst, src1, imm64 } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src1).unwrap_i64() & imm64 as i64)
        )
    }
    I64Or { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a | b));
    }
    I64OrImm { dst, src1, imm64 } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src1).unwrap_i64() | imm64 as i64)
        )
    }
    I64Xor { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a ^ b));
    }
    I64XorImm { dst, src1, imm64 } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src1).unwrap_i64() ^ imm64 as i64)
        )
    }
    I64Shl { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_shl(b as u32)));
    }
    I64ShlImm { dst, src1, imm64 } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src1).unwrap_i64().wrapping_shl(imm64 as u32))
        )
    }
    I64ShrS { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.wrapping_shr(b as u32)));
    }
    I64ShrSImm { dst, src1, imm64 } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src1).unwrap_i64().wrapping_shr(imm64 as u32))
        )
    }
    I64ShrU { dst, src1, src2 } => {
        let (a, b) = (
            get!(src1).unwrap_i64() as u64,
            get!(src2).unwrap_i64() as u32,
        );
        set!(dst, InterpreterValue::i64(a.wrapping_shr(b) as i64));
    }
    I64ShrUImm { dst, src1, imm64 } => {
        set!(
            dst,
            InterpreterValue::i64(
                (get!(src1).unwrap_i64() as u64).wrapping_shr(imm64 as u32) as i64
            )
        )
    }
    I64RotL { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.rotate_left(b as u32)));
    }
    I64RotR { dst, src1, src2 } => {
        let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
        set!(dst, InterpreterValue::i64(a.rotate_right(b as u32)));
    }
    I64Clz { dst, src } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src).unwrap_i64().leading_zeros() as i64)
        )
    }
    I64Ctz { dst, src } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src).unwrap_i64().trailing_zeros() as i64)
        )
    }
    I64Popcnt { dst, src } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src).unwrap_i64().count_ones() as i64)
        )
    }
    I64Eqz { dst, src_val } => {
        set!(dst, InterpreterValue::bool(get!(src_val).unwrap_i64() == 0))
    }
    I64Eq { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i64() == get!(src2).unwrap_i64())
        )
    }
    I64Ne { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i64() != get!(src2).unwrap_i64())
        )
    }
    I64LtS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i64() < get!(src2).unwrap_i64())
        )
    }
    I64LtU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i64() as u64) < (get!(src2).unwrap_i64() as u64)
            )
        )
    }
    I64LeS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i64() <= get!(src2).unwrap_i64())
        )
    }
    I64LeU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i64() as u64) <= (get!(src2).unwrap_i64() as u64)
            )
        )
    }
    I64GtS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i64() > get!(src2).unwrap_i64())
        )
    }
    I64GtU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i64() as u64) > (get!(src2).unwrap_i64() as u64)
            )
        )
    }
    I64GeS { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_i64() >= get!(src2).unwrap_i64())
        )
    }
    I64GeU { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(
                (get!(src1).unwrap_i64() as u64) >= (get!(src2).unwrap_i64() as u64)
            )
        )
    }
}
