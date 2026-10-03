fn main() {
    println!("cargo:rustc-check-cfg=cfg(veloc_native_traps)");
    println!("cargo:rerun-if-changed=src/trap/native.c");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap();
    if os == "linux" && env == "gnu" && matches!(arch.as_str(), "x86_64" | "riscv64") {
        cc::Build::new()
            .file("src/trap/native.c")
            .flag_if_supported("-std=c11")
            .compile("veloc_native_traps");
        println!("cargo:rustc-cfg=veloc_native_traps");
    }
}
