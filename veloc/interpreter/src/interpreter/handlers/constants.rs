use super::super::*;

define_register_handlers! {
    // === Constants ===
    Iconst { dst, imm64 } => {
        set!(dst, InterpreterValue::i64(imm64 as i64))
    }
    Fconst { dst, imm64 } => {
        set!(dst, InterpreterValue(imm64))
    }
    Bconst { dst, val } => {
        set!(dst, InterpreterValue::bool(val))
    }
    Vconst { dst, pool_id } => {
        todo!("Vector constants")
    }
}
