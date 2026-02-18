//! Veloc-C: C Language Compiler Frontend
//!
//! This crate provides a C language parser and AST for the Veloc compiler.

#![allow(dead_code)]

pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;

use std::format;
use std::path::Path;

pub use ast::*;
pub use codegen::{CodeGenContext, compile_to_ir};
pub use error::{Error, Result};
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;
pub use veloc_ir::Module;

/// Parse C source code and return the AST
pub fn parse(source: &str) -> Result<TranslationUnit> {
    let lexer = Lexer::new(source);
    let mut parser = Parser::new(lexer);
    parser.parse_translation_unit()
}

/// Parse C source code from a file path
pub fn parse_file(path: &Path) -> Result<TranslationUnit> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| Error::io_error(format!("Failed to read file: {}", e)))?;
    parse(&source)
}
