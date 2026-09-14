//! Compile generated constant contracts against their real Rust owner.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Temp(PathBuf);

/// Exercise the production loader even for inline definition fixtures.
pub fn source(text: &str) -> Result<veloc_opgen::Source, veloc_opgen::Error> {
    let dir = Temp::new("opgen-source").expect("create definition fixture directory");
    let path = dir.join("module.ops");
    fs::write(&path, text).expect("write definition fixture");
    veloc_opgen::Source::load(path).map_err(|error| error.diagnostic)
}

impl Temp {
    pub fn new(prefix: &str) -> std::io::Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "{prefix}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                // Only acquire cleanup ownership after successfully creating the directory.
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

impl std::ops::Deref for Temp {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn check(generated: &veloc_opgen::Generated) -> Result<Output, String> {
    let dependencies = std::env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or("test executable has no parent")?
        .to_path_buf();
    let owner = fs::read_dir(&dependencies)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with("libveloc_types-") && s.ends_with(".rlib"))
        })
        .max_by_key(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
        .ok_or("missing veloc-types test dependency")?;
    let dir = Temp::new("veloc-contract").map_err(|e| e.to_string())?;
    let source = dir.join("check.rs");
    fs::write(&source, format!("#![feature(const_trait_impl, const_cmp)]\n#![allow(dead_code, unused_imports, unused_parens, unused_variables, unreachable_code)]\npub use veloc_types::Type;\npub struct FuncId;\npub struct VectorConst;\nuse veloc_types::TypeInfo;\npub mod types {{ {} }}\n{}\n", generated.types, generated.checks)).map_err(|e| e.to_string())?;
    Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args([
            "--edition=2024",
            "--crate-type=lib",
            "--emit=metadata",
            "--crate-name=contract",
        ])
        .arg(&source)
        .arg("--extern")
        .arg(format!("veloc_types={}", owner.display()))
        .arg("-L")
        .arg(format!("dependency={}", dependencies.display()))
        .arg("-o")
        .arg(dir.join("check.rmeta"))
        .output()
        .map_err(|e| e.to_string())
}
