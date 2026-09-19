//! Build-time compiler for operation definitions.
//!
//! Definitions are checked before Rust generation. This crate does not depend
//! on a runtime IR; the MIR emitter is one consumer of its definition model.

mod emit;
mod generate;
pub use emit::{Artifacts, Decisions, Emit, Equivalences, Options, Target, ValueRules};
mod model;
pub mod rules;
pub mod schema;
mod semantic;
mod source;
mod storage;
pub mod syntax;
pub mod target;
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
