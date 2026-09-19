use std::{
    env, fs,
    path::{Path, PathBuf},
};
use veloc_spec::{Decisions, Emit, Options, Source, Target};

fn load(path: impl AsRef<Path>) -> Source {
    let source = Source::load(path).expect("load compiler definitions");
    for path in source.dependencies() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    source
}

fn main() {
    println!("cargo:rerun-if-changed=../../rustfmt.toml");
    println!("cargo:rerun-if-env-changed=RUSTFMT");
    let dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let legalize_contract = load("defs/legalize.spec")
        .generate(
            &[Emit::Interfaces],
            Options {
                interfaces: Some("crate::passes::lowering::legalize::contracts"),
                ..Default::default()
            },
        )
        .expect("compile legalization contracts");
    let lir = load("../lir/defs/module.spec");
    let rules = load("defs/x86_64/legalize.spec");
    let target = load("defs/x86_64/module.spec");
    let contracts = load("defs/x86_64/instructions.spec");
    let decisions = rules
        .generate(
            &[Emit::Decisions],
            Options {
                decisions: Some(Decisions {
                    definitions: &lir,
                    rust: veloc_spec::rules::DecisionRust {
                        dialect: "lir",
                        function: "decide",
                        opcode: "veloc_lir::GenericOpcode",
                        field: "veloc_lir::FieldValue",
                        result: "Action",
                        value_interface: "ValueRules",
                        value_adapter: "crate::passes::lowering::RewriteContext::replace_values",
                        rewrite: "crate::passes::lowering::LegalizeAction::rewrite",
                        legal_action: "crate::passes::lowering::LegalizeAction::Legal",
                    },
                }),
                ..Default::default()
            },
        )
        .expect("compile legalization decisions");
    let machine = target
        .generate(
            &[Emit::Target],
            Options {
                target: Some(Target {
                    input: Some(("lir", &lir)),
                    arch: "x86_64",
                    context: "crate::target::x86_64::lowering::X86LoweringContext",
                    definitions: &contracts,
                }),
                ..Default::default()
            },
        )
        .expect("compile target definitions");
    let host = contracts
        .generate(
            &[Emit::Interfaces],
            Options {
                interfaces: Some("crate::target::x86_64::emitter::host"),
                ..Default::default()
            },
        )
        .expect("compile encoder host contracts");
    let mut files = Vec::new();
    for (name, text) in [
        (
            "legalize_contract.rs",
            legalize_contract.get(Emit::Interfaces).unwrap(),
        ),
        (
            "legalize_x86_64.rs",
            decisions.get(Emit::Decisions).unwrap(),
        ),
        ("machine_x86_64.rs", machine.get(Emit::Target).unwrap()),
        ("encoding_host.rs", host.get(Emit::Interfaces).unwrap()),
    ] {
        let path = dir.join(name);
        fs::write(&path, text).expect("write codegen artifacts");
        files.push(path);
    }
    veloc_spec::format_rust(&files, Path::new("../../rustfmt.toml"))
        .expect("format codegen artifacts");
}
