use std::env;
use std::fs;
use std::path::PathBuf;
use veloc_isle::compile;

// The same declarations generate the runtime enum and these build-only bindings.
macro_rules! define_generic_opcodes {
    ($($opcode:ident $(=> $semantic:ident)?),* $(,)?) => {
        const BINDINGS: &[(veloc_semantics::BvOp, &str)] = &[
            $($( (veloc_semantics::BvOp::$semantic, stringify!($opcode)), )?)*
        ];
    };
}
include!("defs/generic.rs");

fn generate_lowering() {
    println!("cargo:rerun-if-changed=defs/generic.rs");
    let mut source = String::new();
    for name in ["types", "builtins", "comparisons", "formats", "mir"] {
        let path = format!("../mir/defs/{name}.ops");
        println!("cargo:rerun-if-changed={path}");
        source.push_str(&fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}")));
        source.push('\n');
    }
    let defs = veloc_opgen::parse(&source).expect("valid MIR definitions");
    let code =
        veloc_opgen::generate_lowering(&defs, BINDINGS).expect("valid LIR primitive bindings");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    fs::write(dir.join("mir_lowering.rs"), code).expect("write direct lowering");
}

fn main() {
    generate_lowering();
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
    }
}
