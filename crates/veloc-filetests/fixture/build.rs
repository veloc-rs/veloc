use std::{env, path::PathBuf};

fn main() {
    let source = veloc_spec::Source::load("module.spec").expect("load fixture definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    use veloc_spec::Emit;
    let output = source
        .generate(
            &[
                Emit::Types,
                Emit::TypeRules,
                Emit::Builders,
                Emit::Validator,
                Emit::Opcodes,
                Emit::Instructions,
                Emit::TextParser,
                Emit::TextPrinter,
                Emit::Evaluation,
                Emit::Semantics,
            ],
            veloc_spec::Options::default(),
        )
        .expect("compile fixture MIR");
    let dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let files = output.write(&dir).expect("write fixture artifacts");
    println!("cargo:rerun-if-changed=../../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    veloc_spec::format_rust(&files, std::path::Path::new("../../../rustfmt.toml")).unwrap();
}
