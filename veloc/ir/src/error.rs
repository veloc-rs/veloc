use crate::text::ParseError;
use crate::validator::ValidationError;
use alloc::string::String;
use core::fmt;

/// A specialized Result type for veloc operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during compilation or IR processing.
#[derive(Debug, Clone)]
pub enum Error {
    /// IR validation failed.
    Validation(ValidationError),
    /// IR parsing failed.
    Parse(String),
    /// Generic error message.
    Message(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(v) => write!(f, "Validation error: {}", v),
            Self::Parse(s) => write!(f, "Parse error: {}", s),
            Self::Message(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl From<ValidationError> for Error {
    fn from(v: ValidationError) -> Self {
        Self::Validation(v)
    }
}

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

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Self::Parse(e.0)
    }
}
