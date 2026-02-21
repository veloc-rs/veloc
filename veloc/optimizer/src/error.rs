use core::fmt;

/// 优化器错误类型
#[derive(Debug, Clone)]
pub enum Error {
    UnknownDebugTag(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnknownDebugTag(tag) => {
                write!(f, "Unknown optimization debug tag: '{}'", tag)
            }
        }
    }
}

impl core::error::Error for Error {}
