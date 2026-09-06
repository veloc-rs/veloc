use crate::value::InterpreterValue;
use veloc_mir::Intrinsic;

pub(super) fn execute_intrinsic(id: u16, args: &[InterpreterValue]) -> InterpreterValue {
    use veloc_mir::intrinsic_ids::*;

    let f = |i: usize| args[i].unwrap_f32();
    let d = |i: usize| args[i].unwrap_f64();

    match Intrinsic::from_u16(id) {
        SIN_F32 => InterpreterValue::f32(libm::sinf(f(0))),
        SIN_F64 => InterpreterValue::f64(libm::sin(d(0))),
        COS_F32 => InterpreterValue::f32(libm::cosf(f(0))),
        COS_F64 => InterpreterValue::f64(libm::cos(d(0))),
        POW_F32 => InterpreterValue::f32(libm::powf(f(0), f(1))),
        POW_F64 => InterpreterValue::f64(libm::pow(d(0), d(1))),
        EXP_F32 => InterpreterValue::f32(libm::expf(f(0))),
        EXP_F64 => InterpreterValue::f64(libm::exp(d(0))),
        LOG_F32 => InterpreterValue::f32(libm::logf(f(0))),
        LOG_F64 => InterpreterValue::f64(libm::log(d(0))),
        LOG2_F32 => InterpreterValue::f32(libm::log2f(f(0))),
        LOG2_F64 => InterpreterValue::f64(libm::log2(d(0))),
        LOG10_F32 => InterpreterValue::f32(libm::log10f(f(0))),
        LOG10_F64 => InterpreterValue::f64(libm::log10(d(0))),
        MEMCPY | MEMMOVE | MEMSET => InterpreterValue::none(),
        MEMCMP => InterpreterValue::i32(0),
        FENCE | FENCE_ACQ | FENCE_REL | FENCE_SEQ => InterpreterValue::none(),
        ASSUME => InterpreterValue::none(),
        EXPECT => args[0],
        TRAP => panic!("trap"),
        _ => panic!("Unknown intrinsic: {}", id),
    }
}
