//! IR 文本表示模块
//!
//! 这个模块提供了 Veloc IR 的文本格式支持，包括：
//! - **Printer**: 将 IR 数据结构格式化为可读的文本
//! - **Parser**: 从文本解析回 IR 数据结构
//!
//! # 文本格式示例
//!
//! ```text
//! export function add(i32, i32) -> i32
//! block0(v0: i32, v1: i32):
//!     v2 = iadd.i32 v0, v1
//!     return v2
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use veloc_ir::text::{parse_function, Printer};
//!
//! // 解析 IR 文本
//! let func = parse_function(ir_text).unwrap();
//!
//! // 打印 IR
//! let printer = Printer::new();
//! let output = printer.print_function(&func);
//! ```

pub mod parser;
pub mod printer;

// 导出常用功能
pub use parser::{
    parse_function, parse_module, ParseError, IRFormat, fmt_type, parse_type_ir,
};
pub use printer::{write_function, write_module};

// 单独导出 format 模块，避免命名冲突
pub use parser::format as fmt;

use crate::{Function, Module, Result};
use alloc::string::String;

/// 将模块格式化为字符串
pub fn format_module(module: &Module) -> String {
    module.to_string()
}

/// 将函数格式化为字符串
pub fn format_function(func: &Function) -> String {
    func.to_string()
}

/// 从文本解析模块（便捷函数）
pub fn parse_module_from_str(input: &str) -> Result<Module> {
    parse_module(input)
}

/// 从文本解析函数（便捷函数）
pub fn parse_function_from_str(input: &str) -> Result<Function> {
    parse_function(input)
}

/// 验证 IR 文本的 round-trip 正确性
///
/// 返回 `(原始文本, 重新格式化后的文本, 是否一致)`
pub fn verify_roundtrip(input: &str) -> Result<(String, String, bool)> {
    let func = parse_function(input)?;
    let reformatted = format_function(&func);
    let original_normalized = normalize_ir_text(input);
    let reformatted_normalized = normalize_ir_text(&reformatted);
    let is_equal = original_normalized == reformatted_normalized;
    Ok((original_normalized, reformatted_normalized, is_equal))
}

/// 规范化 IR 文本用于比较
///
/// - 移除多余空白
/// - 统一换行符
/// - 移除注释
fn normalize_ir_text(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with(';'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// IR 文本构建器
///
/// 用于程序化构建 IR 文本，然后解析为 IR 数据结构
pub struct IRTextBuilder {
    output: String,
    indent_level: usize,
}

impl IRTextBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
        }
    }

    /// 添加函数声明
    pub fn function(&mut self, linkage: &str, name: &str, params: &[&str], ret: &str) -> &mut Self {
        self.line(&format::function_decl(linkage, name, params, ret));
        self
    }

    /// 添加基本块
    pub fn block(&mut self, name: &str, params: &[(u32, &str)]) -> &mut Self {
        self.line(&format::block_header(name, params));
        self.indent_level = 1;
        self
    }

    /// 添加指令
    pub fn inst(&mut self, result: Option<u32>, opcode: &str, args: &[&str]) -> &mut Self {
        let line = format::instruction(result, opcode, args);
        self.indented_line(&line);
        self
    }

    /// 添加原始行
    pub fn line(&mut self, text: &str) -> &mut Self {
        self.output.push_str(text);
        self.output.push('\n');
        self
    }

    /// 添加缩进行
    pub fn indented_line(&mut self, text: &str) -> &mut Self {
        for _ in 0..self.indent_level {
            self.output.push_str("    ");
        }
        self.output.push_str(text);
        self.output.push('\n');
        self
    }

    /// 构建并返回文本
    pub fn build(self) -> String {
        self.output
    }

    /// 构建并解析为函数
    pub fn build_function(self) -> Result<Function> {
        parse_function(&self.output)
    }
}

impl Default for IRTextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 格式辅助函数模块
pub mod format {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    /// 格式化函数声明
    pub fn function_decl(linkage: &str, name: &str, params: &[&str], ret: &str) -> String {
        let params_str = params.join(", ");
        format!("{} function {}({}) -> {}", linkage, name, params_str, ret)
    }

    /// 格式化基本块头
    pub fn block_header(name: &str, params: &[(u32, &str)]) -> String {
        let params_str: Vec<String> = params
            .iter()
            .map(|(idx, ty)| format!("v{}: {}", idx, ty))
            .collect();
        format!("{}({}):", name, params_str.join(", "))
    }

    /// 格式化指令
    pub fn instruction(result: Option<u32>, opcode: &str, args: &[&str]) -> String {
        let result_str = result.map(|r| format!("v{} = ", r)).unwrap_or_default();
        let args_str = args.join(", ");
        format!("{}{} {}", result_str, opcode, args_str)
    }

    /// 格式化值引用
    pub fn value_ref(index: u32) -> String {
        format!("v{}", index)
    }

    /// 格式化块引用
    pub fn block_ref(index: u32) -> String {
        format!("block{}", index)
    }

    /// 格式化栈槽引用
    pub fn stackslot_ref(index: u32) -> String {
        format!("ss{}", index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_builder() {
        let mut builder = IRTextBuilder::new();
        let func = builder
            .function("local", "test", &["i32", "i32"], "i32")
            .block("block0", &[(0, "i32"), (1, "i32")])
            .inst(Some(2), "iadd.i32", &["v0", "v1"])
            .inst(None, "return", &["v2"])
            .build_function()
            .unwrap();

        assert_eq!(func.name, "test");
    }

    #[test]
    fn test_format_helpers() {
        let decl = format::function_decl("export", "add", &["i32", "i32"], "i32");
        assert_eq!(decl, "export function add(i32, i32) -> i32");

        let header = format::block_header("block0", &[(0, "i32"), (1, "i32")]);
        assert_eq!(header, "block0(v0: i32, v1: i32):");

        let inst = format::instruction(Some(2), "iadd.i32", &["v0", "v1"]);
        assert_eq!(inst, "v2 = iadd.i32 v0, v1");
    }
}
