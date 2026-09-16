use std::{env, fs, path::PathBuf};

fn main() {
    let source = veloc_opgen::Source::load("defs/x86_64.ops").expect("load encoder types");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("x86_64.rs");
    fs::write(&path, source.data_types().expect("check encoder types")).unwrap();
    veloc_opgen::format_rust(&[path], std::path::Path::new("../../rustfmt.toml"))
        .expect("format encoder types");
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
}
