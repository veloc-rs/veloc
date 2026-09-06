use alloc::string::String;
use core::fmt;

use crate::lir::MachineOpcode;
use crate::target::arch::TargetArch;
use veloc_mir::Opcode;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub enum Error {
    Codegen(CodegenError),
    Translate(TranslateError),
    Select(InstructionError),
    Emit(InstructionError),
    Message(String),
}

#[derive(Debug, Clone)]
pub enum CodegenError {
    TargetMachineUnavailable {
        arch: TargetArch,
    },
    MissingEmittedCode {
        function: String,
    },
    UnexpectedRelocation {
        symbol: String,
    },
    TranslatedFunctionNotFound {
        function: String,
    },
    ObjectFileRelocation {
        function: String,
        symbol: String,
        message: String,
    },
    ObjectFileWrite {
        message: String,
    },
    UnsupportedObjectFormat {
        arch: TargetArch,
    },
    Message(String),
}

#[derive(Debug, Clone)]
pub enum TranslateError {
    UnsupportedBinaryOpcode { opcode: Opcode },
    UnsupportedUnaryOpcode { opcode: Opcode },
    Message(String),
}

#[derive(Debug, Clone)]
pub struct InstructionError {
    pub opcode: MachineOpcode,
    pub reason: String,
}

impl InstructionError {
    pub fn new(opcode: MachineOpcode, reason: impl Into<String>) -> Self {
        Self {
            opcode,
            reason: reason.into(),
        }
    }
}

impl Error {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    pub fn codegen(message: impl Into<String>) -> Self {
        Self::Codegen(CodegenError::Message(message.into()))
    }

    pub fn target_machine_unavailable(arch: TargetArch) -> Self {
        Self::Codegen(CodegenError::TargetMachineUnavailable { arch })
    }

    pub fn missing_emitted_code(function: impl Into<String>) -> Self {
        Self::Codegen(CodegenError::MissingEmittedCode {
            function: function.into(),
        })
    }

    pub fn unexpected_relocation(symbol: impl Into<String>) -> Self {
        Self::Codegen(CodegenError::UnexpectedRelocation {
            symbol: symbol.into(),
        })
    }

    pub fn translated_function_not_found(function: impl Into<String>) -> Self {
        Self::Codegen(CodegenError::TranslatedFunctionNotFound {
            function: function.into(),
        })
    }

    pub fn object_file_relocation_error(
        function: impl Into<String>,
        symbol: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Codegen(CodegenError::ObjectFileRelocation {
            function: function.into(),
            symbol: symbol.into(),
            message: message.into(),
        })
    }

    pub fn object_file_write_error(message: impl Into<String>) -> Self {
        Self::Codegen(CodegenError::ObjectFileWrite {
            message: message.into(),
        })
    }

    pub fn unsupported_object_format(arch: TargetArch) -> Self {
        Self::Codegen(CodegenError::UnsupportedObjectFormat { arch })
    }

    pub fn translate(message: impl Into<String>) -> Self {
        Self::Translate(TranslateError::Message(message.into()))
    }

    pub fn unsupported_binary_opcode(opcode: Opcode) -> Self {
        Self::Translate(TranslateError::UnsupportedBinaryOpcode { opcode })
    }

    pub fn unsupported_unary_opcode(opcode: Opcode) -> Self {
        Self::Translate(TranslateError::UnsupportedUnaryOpcode { opcode })
    }

    pub fn select(opcode: MachineOpcode, reason: impl Into<String>) -> Self {
        Self::Select(InstructionError::new(opcode, reason))
    }

    pub fn emit(opcode: MachineOpcode, reason: impl Into<String>) -> Self {
        Self::Emit(InstructionError::new(opcode, reason))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Codegen(err) => write!(f, "Codegen error: {}", err),
            Error::Translate(err) => write!(f, "Translation error: {}", err),
            Error::Select(err) => write!(f, "Instruction selection failed for opcode: {}", err),
            Error::Emit(err) => write!(f, "Instruction emission failed for opcode: {}", err),
            Error::Message(s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::TargetMachineUnavailable { arch } => write!(
                f,
                "failed to create target machine for requested architecture `{}`",
                arch.name()
            ),
            CodegenError::MissingEmittedCode { function } => {
                write!(
                    f,
                    "missing emitted code for compiled function `{}`",
                    function
                )
            }
            CodegenError::UnexpectedRelocation { symbol } => write!(
                f,
                "raw code emission cannot resolve external symbol relocation to `{}`",
                symbol
            ),
            CodegenError::TranslatedFunctionNotFound { function } => {
                write!(f, "translated function not found: `{}`", function)
            }
            CodegenError::ObjectFileRelocation {
                function,
                symbol,
                message,
            } => write!(
                f,
                "failed to add relocation for `{}` -> `{}`: {}",
                function, symbol, message
            ),
            CodegenError::ObjectFileWrite { message } => {
                write!(f, "failed to write object file: {}", message)
            }
            CodegenError::UnsupportedObjectFormat { arch } => write!(
                f,
                "object file output is not supported for target architecture `{}`",
                arch.name()
            ),
            CodegenError::Message(s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslateError::UnsupportedBinaryOpcode { opcode } => {
                write!(f, "unsupported binary opcode: {:?}", opcode)
            }
            TranslateError::UnsupportedUnaryOpcode { opcode } => {
                write!(f, "unsupported unary opcode: {:?}", opcode)
            }
            TranslateError::Message(s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Display for InstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} - {}", self.opcode, self.reason)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(feature = "std")]
impl std::error::Error for CodegenError {}

#[cfg(feature = "std")]
impl std::error::Error for TranslateError {}

#[cfg(feature = "std")]
impl std::error::Error for InstructionError {}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Self::Message(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self::Message(s.into())
    }
}

impl From<CodegenError> for Error {
    fn from(err: CodegenError) -> Self {
        Self::Codegen(err)
    }
}

impl From<TranslateError> for Error {
    fn from(err: TranslateError) -> Self {
        Self::Translate(err)
    }
}
