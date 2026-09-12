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

pub use generate::Plan;
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
    pub checks: String,
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
mod fixtures;

pub mod interfaces;
/// Format generated Rust files with the workspace's rustfmt configuration.
///
/// Only the supplied files are formatted; module declarations are not followed.
/// The workspace toolchain includes rustfmt. Missing tools and invalid generated
/// syntax are reported instead of silently leaving unformatted artifacts.
pub fn format_rust(files: &[std::path::PathBuf], config: &std::path::Path) -> std::io::Result<()> {
    use std::{io, process::Command};
    if files.is_empty() {
        return Ok(());
    }
    let output = Command::new(std::env::var_os("RUSTFMT").unwrap_or_else(|| "rustfmt".into()))
        .arg("--config-path")
        .arg(config.canonicalize()?)
        .args(["--config", "skip_children=true"])
        .args(files)
        .output()
        .map_err(|error| io::Error::new(error.kind(), format!("failed to run rustfmt: {error}")))?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "rustfmt failed ({}):\n{}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )));
    }
    Ok(())
}
