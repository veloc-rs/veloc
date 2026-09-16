use std::env;
use std::fs;
use std::path::PathBuf;
use veloc_isle::target::compile;

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let mut rust_files = Vec::new();
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

        let contracts = veloc_opgen::Source::load(isle_dir.join("instructions.ops"))
            .expect("load target instruction definitions");
        let host_code = contracts
            .interfaces("crate::target::x86_64::emitter::host")
            .expect("check encoding host contracts");
        for path in contracts.dependencies() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let output = match compile(&combined_input, arch, &contracts) {
            Ok(out) => out,
            Err(e) => {
                // e 已经是经过 miette 格式化的 Debug 输出（字符串）
                // 在 panic 中直接使用它，或者去掉引号前缀
                panic!("\n\nISLE 编译失败:\n{}\n", e);
            }
        };

        let out_dir = env::var_os("OUT_DIR").map(PathBuf::from).unwrap();
        let host_path = out_dir.join("encoding_host.rs");
        fs::write(&host_path, host_code).expect("write encoding host contracts");
        rust_files.push(host_path);
        let dest_path = out_dir.join(format!("isle_{}.rs", arch));

        fs::write(&dest_path, output).expect("Failed to write generated file");
        rust_files.push(dest_path);
    }
    veloc_opgen::format_rust(&rust_files, std::path::Path::new("../../rustfmt.toml"))
        .expect("format generated codegen definitions");
}
