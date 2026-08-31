use crate::host::HostFuncId;
use alloc::string::String;
use veloc_ir::{FuncId, ModuleId};

pub type Result<T> = core::result::Result<T, Error>;

/// 解释器执行错误
#[derive(Debug, Clone)]
pub enum Error {
    /// 内存访问越界
    OutOfBounds,
    /// 执行了 unreachable 指令
    Unreachable,
    /// 解释器的固定栈空间不足
    StackOverflow,
    /// Referenced module does not exist.
    InvalidModule(ModuleId),
    /// Referenced function does not exist in its module.
    InvalidFunction { module: ModuleId, func: FuncId },
    /// A builder was finished while an import was still unresolved.
    UnresolvedImport { module: ModuleId, func: FuncId },
    /// A link operation was attempted on a defined function.
    ExpectedImport { module: ModuleId, func: FuncId },
    /// Referenced host function does not exist.
    InvalidHostFunction(HostFuncId),
    /// A host call does not match the function's signature.
    InvalidHostCall,
    /// An import's signature does not match the registered host function.
    HostSignatureMismatch {
        module: ModuleId,
        func: FuncId,
        host: HostFuncId,
    },
    /// Source and target function signatures differ.
    SignatureMismatch {
        module: ModuleId,
        func: FuncId,
        target_module: ModuleId,
        target_func: FuncId,
    },
    /// An indirect-call value is not a function reference owned by this program.
    InvalidFunctionReference,
    /// 通用错误消息
    Message(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfBounds => f.write_str("memory access out of bounds"),
            Self::Unreachable => f.write_str("unreachable instruction executed"),
            Self::StackOverflow => f.write_str("interpreter stack overflow"),
            Self::InvalidModule(module) => write!(f, "invalid module {module:?}"),
            Self::InvalidFunction { module, func } => {
                write!(f, "invalid function {func:?} in {module:?}")
            }
            Self::UnresolvedImport { module, func } => {
                write!(f, "unresolved import {func:?} in {module:?}")
            }
            Self::ExpectedImport { module, func } => {
                write!(f, "function {func:?} in {module:?} is not an import")
            }
            Self::InvalidHostFunction(host) => {
                write!(f, "invalid host function {host:?}")
            }
            Self::InvalidHostCall => f.write_str("invalid host function call"),
            Self::HostSignatureMismatch { module, func, host } => write!(
                f,
                "signature mismatch linking {module:?}/{func:?} to {host:?}"
            ),
            Self::SignatureMismatch {
                module,
                func,
                target_module,
                target_func,
            } => write!(
                f,
                "signature mismatch linking {module:?}/{func:?} to \
                 {target_module:?}/{target_func:?}"
            ),
            Self::InvalidFunctionReference => f.write_str("invalid function reference"),
            Self::Message(message) => f.write_str(message),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
