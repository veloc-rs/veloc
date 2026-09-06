use alloc::string::String;
use core::fmt;

use crate::MachineOpcode;

pub type Result<T> = core::result::Result<T, DecodeError>;

/// An instruction does not conform to its LIR operand schema.
#[derive(Debug, Clone)]
pub struct DecodeError {
    pub opcode: MachineOpcode,
    pub reason: String,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIR instruction {:?}: {}",
            self.opcode, self.reason
        )
    }
}

impl core::error::Error for DecodeError {}
