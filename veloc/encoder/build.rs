use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut files = Vec::new();
    for arch in ["x86_64", "riscv64"] {
        let source = veloc_spec::Source::load(format!("defs/{arch}.spec"))
            .expect("load encoder definitions");
        for path in source.dependencies() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let artifacts = source
            .generate(
                &[veloc_spec::Emit::DataTypes],
                veloc_spec::Options::default(),
            )
            .expect("check encoder types");
        let path = dir.join(format!("{arch}.rs"));
        fs::write(&path, artifacts.get(veloc_spec::Emit::DataTypes).unwrap()).unwrap();
        files.push(path);
    }
    veloc_spec::format_rust(&files, Path::new("../../rustfmt.toml")).expect("format encoder types");
}
