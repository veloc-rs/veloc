use std::{
    env,
    path::{Path, PathBuf},
};
use veloc_spec::{Emit, Options, Source};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source = Source::load("defs/module.spec").expect("load LIR definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let output = source
        .generate(
            &[
                Emit::Instructions,
                Emit::TypeRules,
                Emit::Types,
                Emit::Semantics,
            ],
            Options::default(),
        )
        .expect("compile LIR definitions");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let files = output.write(&dir).expect("write LIR artifacts");
    veloc_spec::format_rust(&files, Path::new("../../rustfmt.toml")).expect("format LIR artifacts");
}
