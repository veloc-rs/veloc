pub mod ast;
pub mod compiler;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub fn compile(input: &str, arch: &str) -> Result<String, String> {
    compiler::compile(input, arch)
}
pub use parser::parse;
