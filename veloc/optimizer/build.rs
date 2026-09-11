use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source =
        veloc_opgen::Source::load("../mir/defs/module.ops").expect("load operation definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let generated = source.compile().expect("compile operation definitions");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    fs::write(dir.join("evaluation.rs"), generated.evaluation).unwrap();
    // Only included by offline tests/examples, not the optimizer library.
    fs::write(dir.join("semantics.rs"), generated.semantics).unwrap();
    veloc_opgen::format_rust(
        &[dir.join("evaluation.rs"), dir.join("semantics.rs")],
        std::path::Path::new("../../rustfmt.toml"),
    )
    .expect("format generated optimizer definitions");
}
