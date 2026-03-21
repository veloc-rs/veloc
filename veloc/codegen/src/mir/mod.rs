//! Machine IR (MIR) - 机器无关的中间表示
//!
//! 这是 GlobalISel 流程中的核心数据结构，
//! 在 IRTranslator 生成后，经过 Legalizer、RegBankSelect、InstructionSelect 等阶段处理

// Machine IR (MIR) - 机器相关的中间表示
//
// 此目录仅包含 MIR 本身的定义（指令、基本块、函数）。

pub mod extra;
pub mod function;
pub mod instr;
pub mod module;
pub mod symbol;
pub mod use_def;

pub use crate::regalloc::regbank_select::RegisterBank;
pub use extra::*;
pub use function::*;
pub use instr::*;
pub use module::*;
pub use symbol::*;
pub use use_def::UseDefChain;
pub use veloc_ir::Value as ValueId;
