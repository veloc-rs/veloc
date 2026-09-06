use std::{env, fs, path::PathBuf};

fn main() {
    let mut source = String::new();
    for path in [
        "../../../veloc/mir/defs/types.ops",
        "../../../veloc/mir/defs/builtins.ops",
        "../../../veloc/mir/defs/comparisons.ops",
        "../../../veloc/mir/defs/formats.ops",
        "../../../veloc/mir/defs/mir.ops",
        "extra.ops",
    ] {
        println!("cargo:rerun-if-changed={path}");
        source.push_str(&fs::read_to_string(path).unwrap());
        source.push('\n');
    }
    let output = veloc_opgen::compile_mir(&source).expect("compile fixture MIR");
    let definitions = veloc_opgen::parse(&source).unwrap();
    let lowering = veloc_opgen::generate_lowering(
        &definitions,
        &[
            (veloc_semantics::BvOp::Sub, "G_SUB"),
            (veloc_semantics::BvOp::Add, "G_ADD"),
            (veloc_semantics::BvOp::UDiv, "G_UDIV"),
        ],
    )
    .unwrap();
    let dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut files = Vec::new();
    for (name, text) in [
        ("encoding.rs", output.encoding),
        ("builtins.rs", output.builtins),
        ("scalars.rs", output.scalars),
        ("formats.rs", output.formats),
        ("type_rules.rs", output.types),
        ("validation.rs", output.validation),
        ("opcodes.rs", output.opcodes),
        ("instructions.rs", output.instructions),
        ("text_parser.rs", output.text_parser),
        ("text_printer.rs", output.text_printer),
        ("evaluation.rs", output.evaluation),
        ("semantics.rs", output.semantics),
        ("lowering.rs", lowering),
    ] {
        let path = dir.join(name);
        fs::write(&path, text).unwrap();
        files.push(path);
    }
    println!("cargo:rerun-if-changed=../../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    veloc_opgen::format_rust(&files, std::path::Path::new("../../../rustfmt.toml")).unwrap();
}
