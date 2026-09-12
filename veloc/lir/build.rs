use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source = veloc_opgen::Source::load("defs/module.ops").expect("load LIR definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let generated = source.compile().expect("compile LIR definitions");
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("instructions.rs");
    fs::write(&path, generated.instructions).expect("write generated LIR");
    let rules = path.with_file_name("type_rules.rs");
    fs::write(&rules, generated.type_rules).expect("write generated type rules");
    let types = path.with_file_name("types.rs");
    fs::write(&types, generated.types).expect("write generated type declarations");
    let semantics = path.with_file_name("semantics.rs");
    fs::write(&semantics, generated.semantics).expect("write offline semantics");
    veloc_opgen::format_rust(
        &[path, rules, semantics, types],
        std::path::Path::new("../../rustfmt.toml"),
    )
    .expect("format generated LIR");
}
