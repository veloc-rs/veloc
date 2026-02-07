use crate::vm::TrapCode;
use alloc::string::String;
use core::fmt;

/// A specialized Result type for veloc_wasm operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during WebAssembly module processing and execution.
#[derive(Debug)]
pub enum Error {
    /// WebAssembly parsing or validation error.
    Wasm(wasmparser::BinaryReaderError),
    /// Error during compilation to native code.
    Compile(String),
    /// Error during ELF loading or relocation.
    Link(elf_loader::Error),
    /// Runtime trap.
    Trap(TrapCode),
    /// Export not found.
    ExportNotFound(String),
    /// Import not found.
    ImportNotFound { module: String, field: String },
    /// Incompatible import type.
    IncompatibleImport {
        module: String,
        field: String,
        expected: String,
        actual: String,
    },
    /// Memory allocation error.
    Memory(String),
    /// Generic error message.
    Message(String),
}

impl From<wasmparser::BinaryReaderError> for Error {
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        Error::Wasm(e)
    }
}

impl From<elf_loader::Error> for Error {
    fn from(e: elf_loader::Error) -> Self {
        Error::Link(e)
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Message(s)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Wasm(e) => write!(f, "Wasm error: {}", e),
            Error::Compile(s) => write!(f, "Compile error: {}", s),
            Error::Link(e) => write!(f, "Link error: {}", e),
            Error::Trap(code) => write!(f, "Runtime trap: {:?}", code),
            Error::ExportNotFound(name) => write!(f, "Export not found: {}", name),
            Error::ImportNotFound { module, field } => {
                write!(f, "Import not found: {}.{}", module, field)
            }
            Error::IncompatibleImport {
                module,
                field,
                expected,
                actual,
            } => write!(
                f,
                "Incompatible import {}.{}: expected {}, actual {}",
                module, field, expected, actual
            ),
            Error::Memory(s) => write!(f, "Memory error: {}", s),
            Error::Message(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Wasm(e) => Some(e),
            _ => None,
        }
    }
}
