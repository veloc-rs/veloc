use std::{env, fs, path::PathBuf};

fn main() {
    let source = veloc_opgen::Source::load("defs/types.ops").expect("load type interfaces");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    let code = source
        .interfaces("veloc_types::traits")
        .expect("check type interfaces");
    let file = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("traits.rs");
    fs::write(&file, code).expect("write type interfaces");
    veloc_opgen::format_rust(&[file], std::path::Path::new("../../rustfmt.toml"))
        .expect("format type interfaces");
}
