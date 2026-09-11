//! Build-time compiler for operation definitions.
//!
//! Definitions are checked before Rust generation. This crate does not depend
//! on a runtime IR; the MIR emitter is one consumer of its definition model.

mod builtin_gen;
mod builtins;
mod comparisons;
mod constraints;
mod control;
mod encoding;
mod evaluate;
mod format;
mod generate;
mod lowering;
mod memory;
mod model;
mod ownership;
mod packing;
mod records;
mod semantic;
mod source;
mod storage;
mod syntax;
mod text;
mod type_expr;
mod type_gen;
mod type_rules;
mod type_set;
mod types;

pub use format::format_rust;
pub use lowering::generate_lowering;
pub use model::Definitions;
pub use source::{Source, SourceError};

/// A diagnostic in the definition source (one-based line and column).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl Error {
    fn at(source: &str, offset: usize, message: impl Into<String>) -> Self {
        let prefix = &source.as_bytes()[..offset.min(source.len())];
        Self {
            line: 1 + prefix.iter().filter(|&&b| b == b'\n').count(),
            column: 1 + prefix.iter().rev().take_while(|&&b| b != b'\n').count(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for Error {}

/// Generated MIR, optimizer and offline artifacts; callers choose which to write.
#[derive(Default)]
pub struct Generated {
    pub types: String,
    pub type_rules: String,
    pub validation: String,
    pub evaluation: String,
    pub semantics: String,
    pub opcodes: String,
    pub instructions: String,
    pub builders: String,
    pub text_parser: String,
    pub text_printer: String,
}

/// Parse and check a definition unit, including cross-record references.
pub fn parse(source: &str) -> Result<Definitions, Error> {
    model::parse(source)
}

/// Compile checked operations using the declared storage strategy.
pub fn compile(source: &str) -> Result<Generated, Error> {
    let definitions = parse(source)?;
    generate::generate(&definitions, source)
}

#[cfg(test)]
mod fixtures {
    use super::*;

    const BUILTINS: &str = concat!(
        include_str!("../../defs/types.ops"),
        "\n",
        include_str!("../../defs/builtins.ops"),
        "\n",
        include_str!("../../defs/comparisons.ops")
    );

    pub fn builtins() -> builtins::Builtins {
        super::parse(BUILTINS).unwrap().builtins
    }

    pub fn types() -> types::Types {
        super::parse(BUILTINS).unwrap().types
    }

    pub fn set(expression: &str) -> type_set::TypeSet {
        super::parse(&format!(
            "{BUILTINS}\nclass TestSet {{ members: [{expression}] }}"
        ))
        .unwrap()
        .types
        .classes
        .remove("TestSet")
        .unwrap()
    }

    pub fn parse(source: &str) -> Result<Definitions, Error> {
        super::parse(&format!("{BUILTINS}\n{source}"))
    }

    pub fn compile(source: &str) -> Result<Generated, Error> {
        super::compile(&format!("{BUILTINS}\n{source}"))
    }
}
