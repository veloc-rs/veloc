use std::env;
use std::fs;
use std::path::PathBuf;
use veloc_isle::target::compile;

fn generate_lowering() -> PathBuf {
    let source = veloc_opgen::Source::load("../mir/defs/module.ops").expect("load MIR definitions");
    let lir_source =
        veloc_opgen::Source::load("../lir/defs/module.ops").expect("load LIR definitions");
    for path in source.dependencies().chain(lir_source.dependencies()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let defs = source.parse().expect("check MIR definitions");
    let lir = lir_source.parse().expect("check LIR definitions");
    let mut dialects = veloc_isle::rules::Dialects::default();
    dialects
        .insert("mir", &defs)
        .expect("register MIR contracts");
    dialects
        .insert("lir", &lir)
        .expect("register LIR contracts");
    let rules = "rules/mir.rules";
    println!("cargo:rerun-if-changed={rules}");
    let input = fs::read_to_string(rules).expect("read MIR lowering rules");
    let mut program =
        veloc_isle::rules::Program::compile(&input, &dialects).expect("check MIR lowering rules");
    program
        .infer_primitives(&dialects, "mir", "lir")
        .expect("unambiguous semantic mappings");
    let mut code = program
        .rust(veloc_isle::rules::Rust {
            function: "lower",
            context: "Context",
            source: ("mir", "veloc_mir::Opcode"),
            target: ("lir", "veloc_lir::GenericOpcode"),
        })
        .expect("generate MIR lowering");
    // Invoke OpSpec's generated constructors: physical field order and omitted
    // optional operands must never be reconstructed by the rule compiler.
    use std::fmt::Write;
    let operations: std::collections::BTreeMap<_, _> =
        lir.operations().map(|op| (op.name.clone(), op)).collect();
    code.push_str("fn build(opcode: GenericOpcode, results: &[Reg], inputs: &[Reg]) -> MachineInst { match opcode {\n");
    for target in program.targets() {
        let name = target.strip_prefix("lir.").expect("LIR constructor");
        let op = &operations[name];
        let signature = op.signature.as_ref().expect("checked value signature");
        let builder = op
            .constructor
            .as_ref()
            .expect("operand storage constructor");
        let args = (0..signature.results.len())
            .map(|i| format!("Writable(results[{i}])"))
            .chain((0..signature.inputs.len()).map(|i| format!("inputs[{i}]")))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            code,
            "GenericOpcode::{name} => MachineInst::{builder}({args}),"
        )
        .unwrap();
    }
    code.push_str("_ => unreachable!(\"unbound rule constructor\"),\n} }\n");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let path = dir.join("mir_lowering.rs");
    fs::write(&path, code).expect("write direct lowering");
    path
}

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let mut rust_files = vec![generate_lowering()];
    let arch = "x86_64";
    let isle_dir = PathBuf::from(format!("isle/{}", arch));

    if isle_dir.exists() {
        let mut combined_input = String::new();

        // 加载所有 .isle 文件
        let mut isle_files = Vec::new();
        let mut dirs = vec![isle_dir.clone()];
        while let Some(dir) = dirs.pop() {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        dirs.push(path);
                    } else if path.extension().map_or(false, |ext| ext == "isle") {
                        isle_files.push(path);
                    }
                }
            }
        }
        isle_files.sort();

        for path in isle_files {
            let content = fs::read_to_string(&path).expect("Failed to read ISLE file");
            combined_input.push_str(&content);
            combined_input.push_str("\n\n");
            println!("cargo:rerun-if-changed={}", path.display());
        }

        let output = match compile(&combined_input, arch) {
            Ok(out) => out,
            Err(e) => {
                // e 已经是经过 miette 格式化的 Debug 输出（字符串）
                // 在 panic 中直接使用它，或者去掉引号前缀
                panic!("\n\nISLE 编译失败:\n{}\n", e);
            }
        };

        let out_dir = env::var_os("OUT_DIR").map(PathBuf::from).unwrap();
        let dest_path = out_dir.join(format!("isle_{}.rs", arch));

        fs::write(&dest_path, output).expect("Failed to write generated file");
        rust_files.push(dest_path);
    }
    veloc_opgen::format_rust(&rust_files, std::path::Path::new("../../rustfmt.toml"))
        .expect("format generated codegen definitions");
}
