use std::{env, fs, path::PathBuf};

fn main() {
    let source = veloc_spec::Source::load("defs/types.spec").expect("load type interfaces");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    let code = source
        .generate(
            &[veloc_spec::Emit::Interfaces],
            veloc_spec::Options {
                interfaces: Some("veloc_types::traits"),
                ..Default::default()
            },
        )
        .expect("check type interfaces");
    let file = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("traits.rs");
    fs::write(&file, code.get(veloc_spec::Emit::Interfaces).unwrap())
        .expect("write type interfaces");
    veloc_spec::format_rust(&[file], std::path::Path::new("../../rustfmt.toml"))
        .expect("format type interfaces");
}
