use std::{env, fs, path::PathBuf};

fn main() {
    let source = veloc_spec::Source::load("defs/x86_64.spec").expect("load encoder types");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("x86_64.rs");
    let artifacts = source
        .generate(
            &[veloc_spec::Emit::DataTypes],
            veloc_spec::Options::default(),
        )
        .expect("check encoder types");
    fs::write(&path, artifacts.get(veloc_spec::Emit::DataTypes).unwrap()).unwrap();
    veloc_spec::format_rust(&[path], std::path::Path::new("../../rustfmt.toml"))
        .expect("format encoder types");
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
}
