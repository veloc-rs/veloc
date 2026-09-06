//! Formatting at the artifact boundary, after all Rust files have been emitted.
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Format generated Rust files with the workspace's rustfmt configuration.
///
/// Only the supplied files are formatted; module declarations are not followed.
/// The workspace toolchain includes rustfmt. Missing tools and invalid generated
/// syntax are reported instead of silently leaving unformatted artifacts.
pub fn format_rust(files: &[PathBuf], config: &Path) -> io::Result<()> {
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
