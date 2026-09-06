use super::super::*;

define_register_handlers! {
    // === Conversions ===
    ExtendS { dst, src, ty } => {
        let val = get!(src).unwrap_i64();
        let res = match ty.from {
            Type::I8 => val as i8 as i64,
            Type::I16 => val as i16 as i64,
            Type::I32 => val as i32 as i64,
            _ => panic!("Unsupported ExtendS from_ty: {:?}", ty.from),
        };
        set!(
            dst,
            if ty.to == Type::I32 {
                InterpreterValue::i32(res as i32)
            } else {
                InterpreterValue::i64(res)
            }
        );
    }
    ExtendU { dst, src, ty } => {
        let val = get!(src).unwrap_i64();
        let res = match ty.from {
            Type::I8 => (val as u8) as u64 as i64,
            Type::I16 => (val as u16) as u64 as i64,
            Type::I32 => (val as u32) as u64 as i64,
            _ => panic!("Unsupported ExtendU from_ty: {:?}", ty.from),
        };
        set!(
            dst,
            if ty.to == Type::I32 {
                InterpreterValue::i32(res as i32)
            } else {
                InterpreterValue::i64(res)
            }
        );
    }
    Wrap { dst, src, ty } => {
        let val = get!(src).unwrap_i64();
        let res = match ty.to {
            Type::I8 => val as i8 as i64,
            Type::I16 => val as i16 as i64,
            Type::I32 => val as i32 as i64,
            _ => val,
        };
        set!(
            dst,
            if ty.to == Type::I32 {
                InterpreterValue::i32(res as i32)
            } else {
                InterpreterValue::i64(res)
            }
        );
    }

    I32TruncF32S { dst, src } => {
        set!(dst, InterpreterValue::i32(get!(src).unwrap_f32() as i32))
    }
    I32TruncF32U { dst, src } => {
        set!(
            dst,
            InterpreterValue::i32(get!(src).unwrap_f32() as u32 as i32)
        )
    }
    I32TruncF64S { dst, src } => {
        set!(dst, InterpreterValue::i32(get!(src).unwrap_f64() as i32))
    }
    I32TruncF64U { dst, src } => {
        set!(
            dst,
            InterpreterValue::i32(get!(src).unwrap_f64() as u32 as i32)
        )
    }
    I64TruncF32S { dst, src } => {
        set!(dst, InterpreterValue::i64(get!(src).unwrap_f32() as i64))
    }
    I64TruncF32U { dst, src } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src).unwrap_f32() as u64 as i64)
        )
    }
    I64TruncF64S { dst, src } => {
        set!(dst, InterpreterValue::i64(get!(src).unwrap_f64() as i64))
    }
    I64TruncF64U { dst, src } => {
        set!(
            dst,
            InterpreterValue::i64(get!(src).unwrap_f64() as u64 as i64)
        )
    }

    F32DemoteF64 { dst, src } => {
        set!(dst, InterpreterValue::f32(get!(src).unwrap_f64() as f32))
    }
    F64PromoteF32 { dst, src } => {
        set!(dst, InterpreterValue::f64(get!(src).unwrap_f32() as f64))
    }
    I32TruncSatF32S { dst, src } => {
        let val = get!(src).unwrap_f32();
        set!(
            dst,
            InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
        );
    }
    I32TruncSatF32U { dst, src } => {
        let val = get!(src).unwrap_f32();
        set!(
            dst,
            InterpreterValue::i32(if val.is_nan() || val < 0.0 {
                0
            } else {
                val as u32
            } as i32)
        );
    }
    I32TruncSatF64S { dst, src } => {
        let val = get!(src).unwrap_f64();
        set!(
            dst,
            InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
        );
    }
    I32TruncSatF64U { dst, src } => {
        let val = get!(src).unwrap_f64();
        set!(
            dst,
            InterpreterValue::i32(if val.is_nan() || val < 0.0 {
                0
            } else {
                val as u32
            } as i32)
        );
    }
    I64TruncSatF32S { dst, src } => {
        let val = get!(src).unwrap_f32();
        set!(
            dst,
            InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
        );
    }
    I64TruncSatF32U { dst, src } => {
        let val = get!(src).unwrap_f32();
        set!(
            dst,
            InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                0
            } else {
                val as u64
            } as i64)
        );
    }
    I64TruncSatF64S { dst, src } => {
        let val = get!(src).unwrap_f64();
        set!(
            dst,
            InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
        );
    }
    I64TruncSatF64U { dst, src } => {
        let val = get!(src).unwrap_f64();
        set!(
            dst,
            InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                0
            } else {
                val as u64
            } as i64)
        );
    }
    F32ConvertI32S { dst, src } => {
        set!(dst, InterpreterValue::f32(get!(src).unwrap_i32() as f32))
    }
    F32ConvertI32U { dst, src } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src).unwrap_i32() as u32 as f32)
        )
    }
    F32ConvertI64S { dst, src } => {
        set!(dst, InterpreterValue::f32(get!(src).unwrap_i64() as f32))
    }
    F32ConvertI64U { dst, src } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src).unwrap_i64() as u64 as f32)
        )
    }
    F64ConvertI32S { dst, src } => {
        set!(dst, InterpreterValue::f64(get!(src).unwrap_i32() as f64))
    }
    F64ConvertI32U { dst, src } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src).unwrap_i32() as u32 as f64)
        )
    }
    F64ConvertI64S { dst, src } => {
        set!(dst, InterpreterValue::f64(get!(src).unwrap_i64() as f64))
    }
    F64ConvertI64U { dst, src } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src).unwrap_i64() as u64 as f64)
        )
    }
}
