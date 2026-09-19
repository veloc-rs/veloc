use std::{
    env,
    path::{Path, PathBuf},
};
use veloc_spec::{Emit, Equivalences, Options, Source};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source = Source::load("../mir/defs/module.spec").expect("load optimizer definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let output = source
        .generate(&[Emit::Evaluation, Emit::Semantics], Options::default())
        .expect("compile optimizer definitions");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let mut files = output.write(&dir).expect("write optimizer artifacts");
    let rules = Source::load("defs/equivalences.spec").expect("load equivalence rules");
    for path in rules.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let generated = rules
        .generate(
            &[Emit::Equivalences],
            Options {
                equivalences: Some(Equivalences {
                    definitions: &source,
                    dialect: "mir",
                    opcode: "veloc_mir::Opcode",
                    types: "veloc_mir::Type",
                }),
                ..Default::default()
            },
        )
        .expect("compile equivalence rules");
    files.extend(generated.write(&dir).expect("write equivalence rules"));
    veloc_spec::format_rust(&files, Path::new("../../rustfmt.toml"))
        .expect("format optimizer artifacts");
}
