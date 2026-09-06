use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let mut source = String::new();
    for name in ["types", "builtins", "comparisons", "formats", "mir"] {
        let path = format!("../mir/defs/{name}.ops");
        println!("cargo:rerun-if-changed={path}");
        source.push_str(&fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}")));
        source.push('\n');
    }
    let generated = veloc_opgen::compile_mir(&source).expect("valid operation definitions");
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
