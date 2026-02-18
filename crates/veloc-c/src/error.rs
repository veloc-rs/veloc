//! Error handling for the C compiler frontend

use core::fmt;
use std::string::String;

/// Result type alias for the C compiler
pub type Result<T> = core::result::Result<T, Error>;

/// Error types for the C compiler frontend
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// I/O error
    Io(String),
    /// Lexical error
    Lexical {
        message: String,
        line: u32,
        column: u32,
    },
    /// Syntax error
    Syntax {
        message: String,
        line: u32,
        column: u32,
    },
    /// Semantic error
    Semantic {
        message: String,
        line: u32,
        column: u32,
    },
}

impl Error {
    /// Create an I/O error
    pub fn io_error(message: impl Into<String>) -> Self {
        Error::Io(message.into())
    }

    /// Create a lexical error
    pub fn lexical(message: impl Into<String>, line: u32, column: u32) -> Self {
        Error::Lexical {
            message: message.into(),
            line,
            column,
        }
    }

    /// Create a syntax error
    pub fn syntax(message: impl Into<String>, line: u32, column: u32) -> Self {
        Error::Syntax {
            message: message.into(),
            line,
            column,
        }
    }

    /// Create a semantic error
    pub fn semantic(message: impl Into<String>, line: u32, column: u32) -> Self {
        Error::Semantic {
            message: message.into(),
            line,
            column,
        }
    }

    /// Get the line number where the error occurred
    pub fn line(&self) -> u32 {
        match self {
            Error::Io(_) => 0,
            Error::Lexical { line, .. } => *line,
            Error::Syntax { line, .. } => *line,
            Error::Semantic { line, .. } => *line,
        }
    }

    /// Get the column number where the error occurred
    pub fn column(&self) -> u32 {
        match self {
            Error::Io(_) => 0,
            Error::Lexical { column, .. } => *column,
            Error::Syntax { column, .. } => *column,
            Error::Semantic { column, .. } => *column,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(msg) => write!(f, "I/O error: {}", msg),
            Error::Lexical {
                message,
                line,
                column,
            } => {
                write!(f, "Lexical error at {}:{}: {}", line, column, message)
            }
            Error::Syntax {
                message,
                line,
                column,
            } => {
                write!(f, "Syntax error at {}:{}: {}", line, column, message)
            }
            Error::Semantic {
                message,
                line,
                column,
            } => {
                write!(f, "Semantic error at {}:{}: {}", line, column, message)
            }
        }
    }
}

impl std::error::Error for Error {}
