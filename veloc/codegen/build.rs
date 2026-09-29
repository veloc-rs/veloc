use std::{
    env, fs,
    path::{Path, PathBuf},
};
use veloc_spec::{Decisions, Emit, Options, Source, Target};

fn load(path: impl AsRef<Path>) -> Source {
    let path = path.as_ref();
    let source = Source::load(path).unwrap_or_else(|err| panic!("load {}: {err}", path.display()));
    for dependency in source.dependencies() {
        println!("cargo:rerun-if-changed={}", dependency.display());
    }
    source
}

/// All generated files use the same write/format path and retain source context
/// in build errors. Architecture-specific code only supplies generator options.
struct Generator {
    out: PathBuf,
    files: Vec<PathBuf>,
}
impl Generator {
    fn emit(&mut self, source: &Source, kind: Emit, options: Options<'_>, name: &str) {
        let artifacts = source
            .generate(&[kind], options)
            .unwrap_or_else(|err| panic!("generate {name}: {err}"));
        let path = self.out.join(name);
        fs::write(&path, artifacts.get(kind).expect("requested artifact"))
            .unwrap_or_else(|err| panic!("write {}: {err}", path.display()));
        self.files.push(path);
    }

    fn legalizer(&mut self, arch: &str, lir: &Source) {
        let source = load(format!("defs/{arch}/legalize.spec"));
        self.emit(
            &source,
            Emit::Decisions,
            Options {
                decisions: Some(Decisions {
                    definitions: lir,
                    rust: veloc_spec::rules::DecisionRust {
                        dialect: "lir",
                        function: "program",
                        opcode: "veloc_lir::GenericOpcode",
                        field: "veloc_lir::FieldValue",
                        value: "veloc_lir::Reg",
                        runtime: "crate::passes::lowering::legalize::vm",
                    },
                }),
                ..Default::default()
            },
            &format!("legalize_{arch}.rs"),
        );
    }

    fn machine(&mut self, arch: &str, lir: &Source, context: &str, host: &str) {
        let source = load(format!("defs/{arch}/module.spec"));
        let contracts = load(format!("defs/{arch}/instructions.spec"));
        self.emit(
            &source,
            Emit::Target,
            Options {
                target: Some(Target {
                    input: Some(("lir", lir)),
                    arch,
                    context,
                    definitions: &contracts,
                }),
                ..Default::default()
            },
            &format!("machine_{arch}.rs"),
        );
        self.emit(
            &contracts,
            Emit::Interfaces,
            Options {
                interfaces: Some(host),
                ..Default::default()
            },
            &format!("encoding_host_{arch}.rs"),
        );
    }
}

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let lir = load("../lir/defs/module.spec");
    let mut generator = Generator {
        out: PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR")),
        files: Vec::new(),
    };
    for arch in ["x86_64", "riscv64"] {
        generator.legalizer(arch, &lir);
    }
    generator.machine(
        "x86_64",
        &lir,
        "crate::target::x86_64::lowering::X86LoweringContext",
        "crate::target::x86_64::emitter::host",
    );
    generator.machine(
        "riscv64",
        &lir,
        "crate::target::riscv64::SelectionContext",
        "crate::target::riscv64::emitter::host",
    );
    veloc_spec::format_rust(&generator.files, Path::new("../../rustfmt.toml"))
        .expect("format codegen artifacts");
}
