//! Build-time compiler for operation definitions.
//!
//! Definitions are checked before Rust generation. This crate does not depend
//! on a runtime IR; the MIR emitter is one consumer of its definition model.

mod builtin_gen;
mod builtins;
mod comparisons;
mod constraints;
mod encoding;
mod evaluate;
mod lowering;
mod mir;
mod model;
mod packing;
mod records;
mod semantic;
mod storage;
mod syntax;
mod text;
mod type_expr;
mod type_gen;
mod type_rules;
mod type_set;
mod types;

pub use lowering::generate_lowering;
pub use model::Definitions;

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
pub struct Generated {
    pub encoding: String,
    pub builtins: String,
    pub scalars: String,
    pub formats: String,
    pub types: String,
    pub validation: String,
    pub evaluation: String,
    pub semantics: String,
    pub opcodes: String,
    pub instructions: String,
    pub text_parser: String,
    pub text_printer: String,
}

/// Parse and check a definition unit, including cross-record references.
pub fn parse(source: &str) -> Result<Definitions, Error> {
    model::parse(source)
}

/// Generate the compact MIR's layouts, metadata and ergonomic builders.
pub fn compile_mir(source: &str) -> Result<Generated, Error> {
    let definitions = parse(source)?;
    mir::generate(&definitions, source)
}

#[cfg(test)]
mod fixtures {
    use super::*;

    const BUILTINS: &str = concat!(
        include_str!("../../mir/defs/types.ops"),
        "\n",
        include_str!("../../mir/defs/builtins.ops"),
        "\n",
        include_str!("../../mir/defs/comparisons.ops")
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

    pub fn compile_mir(source: &str) -> Result<Generated, Error> {
        super::compile_mir(&format!("{BUILTINS}\n{source}"))
    }
}
