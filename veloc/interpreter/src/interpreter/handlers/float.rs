use super::super::*;

define_register_handlers! {
    // === F32 Operations ===
    F32Add { dst, src1, src2 } => {
        let lhs = get!(src1).unwrap_f32();
        let rhs = get!(src2).unwrap_f32();
        set!(dst, InterpreterValue::f32(lhs + rhs));
    }
    F32Sub { dst, src1, src2 } => {
        let lhs = get!(src1).unwrap_f32();
        let rhs = get!(src2).unwrap_f32();
        set!(dst, InterpreterValue::f32(lhs - rhs));
    }
    F32Mul { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src1).unwrap_f32() * get!(src2).unwrap_f32())
        )
    }
    F32Div { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src1).unwrap_f32() / get!(src2).unwrap_f32())
        )
    }
    F32Abs { dst, src1 } => {
        set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().abs()))
    }
    F32Neg { dst, src1 } => {
        set!(dst, InterpreterValue::f32(-get!(src1).unwrap_f32()))
    }
    F32Sqrt { dst, src1 } => {
        set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().sqrt()))
    }
    F32Ceil { dst, src1 } => {
        set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().ceil()))
    }
    F32Floor { dst, src1 } => {
        set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().floor()))
    }
    F32Trunc { dst, src1 } => {
        set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().trunc()))
    }
    F32Nearest { dst, src1 } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src1).unwrap_f32().round_ties_even())
        )
    }
    F32Min { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src1).unwrap_f32().min(get!(src2).unwrap_f32()))
        )
    }
    F32Max { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f32(get!(src1).unwrap_f32().max(get!(src2).unwrap_f32()))
        )
    }
    F32CopySign { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f32(
                get!(src1).unwrap_f32().copysign(get!(src2).unwrap_f32())
            )
        )
    }
    F32Eq { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f32() == get!(src2).unwrap_f32())
        )
    }
    F32Ne { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f32() != get!(src2).unwrap_f32())
        )
    }
    F32Lt { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f32() < get!(src2).unwrap_f32())
        )
    }
    F32Le { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f32() <= get!(src2).unwrap_f32())
        )
    }
    F32Gt { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f32() > get!(src2).unwrap_f32())
        )
    }
    F32Ge { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f32() >= get!(src2).unwrap_f32())
        )
    }

    // === F64 Operations ===
    F64Add { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64() + get!(src2).unwrap_f64())
        )
    }
    F64Sub { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64() - get!(src2).unwrap_f64())
        )
    }
    F64Mul { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64() * get!(src2).unwrap_f64())
        )
    }
    F64Div { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64() / get!(src2).unwrap_f64())
        )
    }
    F64Abs { dst, src1 } => {
        set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().abs()))
    }
    F64Neg { dst, src1 } => {
        set!(dst, InterpreterValue::f64(-get!(src1).unwrap_f64()))
    }
    F64Sqrt { dst, src1 } => {
        set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().sqrt()))
    }
    F64Ceil { dst, src1 } => {
        set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().ceil()))
    }
    F64Floor { dst, src1 } => {
        set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().floor()))
    }
    F64Trunc { dst, src1 } => {
        set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().trunc()))
    }
    F64Nearest { dst, src1 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64().round_ties_even())
        )
    }
    F64Min { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64().min(get!(src2).unwrap_f64()))
        )
    }
    F64Max { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(get!(src1).unwrap_f64().max(get!(src2).unwrap_f64()))
        )
    }
    F64CopySign { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::f64(
                get!(src1).unwrap_f64().copysign(get!(src2).unwrap_f64())
            )
        )
    }
    F64Eq { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f64() == get!(src2).unwrap_f64())
        )
    }
    F64Ne { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f64() != get!(src2).unwrap_f64())
        )
    }
    F64Lt { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f64() < get!(src2).unwrap_f64())
        )
    }
    F64Le { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f64() <= get!(src2).unwrap_f64())
        )
    }
    F64Gt { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f64() > get!(src2).unwrap_f64())
        )
    }
    F64Ge { dst, src1, src2 } => {
        set!(
            dst,
            InterpreterValue::bool(get!(src1).unwrap_f64() >= get!(src2).unwrap_f64())
        )
    }
}
