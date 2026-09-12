use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let source = veloc_opgen::Source::load("defs/module.ops").expect("load MIR definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let output = source.compile().expect("compile MIR definitions");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let mut rust_files = Vec::new();
    for (name, text) in [
        ("types.rs", output.types),
        ("host_traits.rs", output.host),
        ("type_rules.rs", output.type_rules),
        ("builders.rs", output.builders),
        ("validation.rs", output.validation),
        ("opcodes.rs", output.opcodes),
        ("instructions.rs", output.instructions),
        ("text_parser.rs", output.text_parser),
        ("text_printer.rs", output.text_printer),
    ] {
        let path = dir.join(name);
        fs::write(&path, text).expect("write generated MIR definitions");
        rust_files.push(path);
    }

    veloc_opgen::format_rust(&rust_files, std::path::Path::new("../../rustfmt.toml"))
        .expect("format generated MIR definitions");
}
