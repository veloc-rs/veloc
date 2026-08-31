use alloc::string::String;

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
    /// 通用错误消息
    Message(String),
}
