use alloc::string::String;
use core::fmt;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub enum Error {
    Codegen(String),
    Message(String),
    Translate(String),
    Select(crate::mir::MachineOpcode, String),
    Emit(crate::mir::MachineOpcode, String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Codegen(s) => write!(f, "Codegen error: {}", s),
            Error::Message(s) => write!(f, "{}", s),
            Error::Translate(s) => write!(f, "Translation error: {}", s),
            Error::Select(op, reason) => write!(
                f,
                "Instruction selection failed for opcode: {:?} - {}",
                op, reason
            ),
            Error::Emit(op, reason) => write!(
                f,
                "Instruction emission failed for opcode: {:?} - {}",
                op, reason
            ),
        }
    }
}
