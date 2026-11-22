use std::{
    env,
    path::{Path, PathBuf},
};
use veloc_spec::{Emit, Optimizations, Options, Source};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source = Source::load("../mir/defs/module.spec").expect("load optimizer definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let rules = Source::load("defs/rules.spec").expect("load optimizer rules");
    for path in rules.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let generated = rules
        .generate(
            &[
                Emit::Semantics,
                Emit::Equivalences,
                Emit::LocalFolds,
                Emit::Rewrites,
            ],
            Options {
                optimizations: Some(Optimizations {
                    definitions: &source,
                    dialect: "mir",
                    rust: "veloc_mir",
                }),
                ..Default::default()
            },
        )
        .expect("compile optimizer rules");
    let files = generated.write(&dir).expect("write optimizer artifacts");
    veloc_spec::format_rust(&files, Path::new("../../rustfmt.toml"))
        .expect("format optimizer artifacts");
}
