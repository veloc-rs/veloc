use std::{env, fs, path::PathBuf};

fn main() {
    let mut source = String::new();
    let mut locations = Vec::new();
    let mut line = 1;
    for path in [
        "defs/types.ops",
        "defs/builtins.ops",
        "defs/formats.ops",
        "defs/mir.ops",
    ] {
        println!("cargo:rerun-if-changed={path}");
        let input = fs::read_to_string(path).unwrap_or_else(|err| panic!("{path}: {err}"));
        let end = line + input.bytes().filter(|&b| b == b'\n').count() + 1;
        locations.push((path, line..end));
        line = end;
        source.push_str(&input);
        source.push('\n');
    }
    let output = veloc_opgen::compile_mir(&source).unwrap_or_else(|err| {
        let (path, lines) = locations
            .iter()
            .find(|(_, lines)| lines.contains(&err.line))
            // A missing closing delimiter can point just beyond the last line.
            .unwrap_or_else(|| locations.last().expect("at least one definition input"));
        panic!(
            "{path}:{}:{}: {}",
            err.line - lines.start + 1,
            err.column,
            err.message
        );
    });
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    for (name, text) in [
        ("encoding.rs", output.encoding),
        ("builtins.rs", output.builtins),
        ("scalars.rs", output.scalars),
        ("formats.rs", output.formats),
        ("type_schemes.rs", output.types),
        ("opcodes.rs", output.opcodes),
        ("instructions.rs", output.instructions),
        ("text_parser.rs", output.text_parser),
        ("text_printer.rs", output.text_printer),
    ] {
        fs::write(dir.join(name), text).expect("write generated MIR definitions");
    }
}
