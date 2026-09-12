use std::{env, fs, path::PathBuf};

fn main() {
    let source = veloc_opgen::Source::load("module.ops").expect("load fixture definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let output = source.compile().expect("compile fixture MIR");
    let dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut files = Vec::new();
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
        ("evaluation.rs", output.evaluation),
        ("semantics.rs", output.semantics),
    ] {
        let path = dir.join(name);
        fs::write(&path, text).unwrap();
        files.push(path);
    }
    println!("cargo:rerun-if-changed=../../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    veloc_opgen::format_rust(&files, std::path::Path::new("../../../rustfmt.toml")).unwrap();
}
