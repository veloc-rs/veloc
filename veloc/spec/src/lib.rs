//! Build-time compiler for operation definitions.
//!
//! Definitions are checked before Rust generation. This crate does not depend
//! on a runtime IR; the MIR emitter is one consumer of its definition model.

mod generate;
mod model;
pub mod schema;
mod semantic;
mod source;
mod storage;
pub mod syntax;
mod text;
mod types;

pub use generate::{Plan, format_rust};
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
    pub fn at(source: &str, offset: usize, message: impl Into<String>) -> Self {
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
    pub host: String,
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

/// Prepare all selected-output contracts without emitting Rust.
pub fn plan(source: &str) -> Result<Plan, Error> {
    Plan::prepare(parse(source)?, source)
}

/// Compile checked operations using the declared storage strategy.
pub fn compile(source: &str) -> Result<Generated, Error> {
    Ok(plan(source)?.generate())
}

#[cfg(test)]
mod fixtures {
    use super::*;

    static BUILTINS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        concat!(
            include_str!("../../defs/types.ops"),
            "\n",
            include_str!("../../defs/builtins.ops"),
            "\n",
            include_str!("../../defs/comparisons.ops")
        )
        .lines()
        .filter(|line| !line.trim_start().starts_with("import "))
        .collect::<Vec<_>>()
        .join("\n")
    });

    pub fn types_policy(name: &str) -> crate::model::records::Policy {
        super::parse(&BUILTINS).unwrap().data.rust.policy(name)
    }

    pub fn types() -> types::Types {
        super::parse(&BUILTINS).unwrap().types
    }

    pub fn set(expression: &str) -> types::TypeSet {
        super::parse(&format!("{}\ntypeset TestSet = {expression};", *BUILTINS))
            .unwrap()
            .types
            .sets
            .remove("TestSet")
            .unwrap()
    }

    pub fn parse(source: &str) -> Result<Definitions, Error> {
        super::parse(&format!("{}\n{source}", *BUILTINS))
    }

    pub fn compile(source: &str) -> Result<Generated, Error> {
        super::compile(&format!("{}\n{source}", *BUILTINS))
    }
}
