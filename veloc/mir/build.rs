use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source = veloc_spec::Source::load("defs/module.spec").expect("load MIR definitions");
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
            ],
            veloc_spec::Options::default(),
        )
        .expect("compile MIR definitions");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let rust_files = output.write(&dir).expect("write generated MIR definitions");

    veloc_spec::format_rust(&rust_files, std::path::Path::new("../../rustfmt.toml"))
        .expect("format generated MIR definitions");
}
