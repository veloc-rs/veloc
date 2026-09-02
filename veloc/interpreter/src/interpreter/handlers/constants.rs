use super::super::*;

define_register_handlers! {
    // === Constants ===
    Iconst { dst, imm64 } => {
        set!(dst, InterpreterValue::i64(imm64 as i64))
    }
    Iconst32 { dst, imm32 } => {
        set!(dst, InterpreterValue(imm32 as u64))
    }
    Fconst { dst, imm64 } => {
        set!(dst, InterpreterValue(imm64))
    }
    Fconst32 { dst, bits32 } => {
        set!(dst, InterpreterValue(bits32 as u64))
    }
    Bconst { dst, val } => {
        set!(dst, InterpreterValue::bool(val))
    }
    Vconst { dst, pool_id } => {
        todo!("Vector constants")
    }
}
