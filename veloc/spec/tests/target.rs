//! Production target definitions from OpSpec expansion through machine emission.
use veloc_spec::target::{Def, parse};

#[test]
fn production_target_contracts_generate_all_consumers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../codegen/defs/x86_64");
    let definitions = veloc_spec::Source::load(root.join("instructions.spec")).unwrap();
    let input = veloc_spec::Source::load(root.join("module.spec")).unwrap();
    let lir = veloc_spec::Source::load(root.join("../../../lir/defs/module.spec")).unwrap();
    let compile = || {
        input
            .generate(
                &[veloc_spec::Emit::Target],
                veloc_spec::Options {
                    target: Some(veloc_spec::Target {
                        input: Some(("lir", &lir)),
                        arch: "x86_64",
                        context: "crate::target::x86_64::lowering::X86LoweringContext",
                        definitions: &definitions,
                    }),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let artifacts = compile();
    let output = &artifacts[veloc_spec::Emit::Target];
    for contract in [
        "pub enum TargetInst",
        "pub enum Feature",
        "pub struct FeatureSet",
        "pub struct CpuModel",
        "REGISTER_VIEWS",
        "RegisterWrite::ZeroExtend",
        "REG_RAX",
        "REG_AL",
        "REG_R15D",
        "register_constraints",
        "pub fn validate",
        "pub fn required_features",
        "TargetInst::X86Popcnt32.required_features()",
        "pub fn write_assembly",
        "pub fn emit",
        "GenericOpcode::",
        "ABI_",
    ] {
        assert!(
            output.contains(contract),
            "missing generated contract: {contract}"
        );
    }
    assert_eq!(output, &compile()[veloc_spec::Emit::Target]);
}

#[test]
fn typed_constants_and_templates_share_rust_style_syntax() {
    use veloc_spec::syntax::{DeclKind, Kind};
    let declarations = veloc_spec::syntax::parse(
        r#"
        template Root(Name: ident, Bits: expr) {
            const Name: Register = Register { bits: Bits, encoding: 0 };
        }
        expand Root(RAX, 64);
        const EAX: RegisterView = RegisterView {
            base: RAX, offset: 0, bits: 32, write: WriteEffect::ZeroExtend,
        };
    "#,
    )
    .unwrap();
    assert_eq!(declarations.len(), 2);
    assert_eq!(declarations[0].name, "RAX");
    let DeclKind::Constant {
        ty,
        value: Some(value),
    } = &declarations[0].kind
    else {
        panic!("expected an initialized typed constant");
    };
    assert!(matches!(&ty.kind, Kind::Name(name) if name == "Register"));
    let Kind::Object(_, fields) = &value.kind else {
        panic!("expected a typed literal")
    };
    assert!(matches!(fields["bits"].kind, Kind::Number(64)));
    assert!(parse("(def-reg RAX (size 64) (hw-enc 0))").is_err());
    assert!(parse("(def-regclass GPR64 (RAX))").is_err());
}

#[test]
fn parse_typed_selection_cases() {
    let input = r#"
        select(n: lir::Add) {
            choose {
                case {
                    replace(n, build(X86Add64(n.lhs, n.rhs)));
                }
            }
        }
    "#;

    let module = parse(input).expect("parse should succeed");
    assert_eq!(module.defs.len(), 1);

    let Def::SelectRule(rule) = &module.defs[0] else {
        panic!("expected select-rule");
    };

    assert_eq!(rule.opcode, "lir::Add");
    assert_eq!(rule.fields.len(), 2);
    assert_eq!(rule.builds.len(), 1);
    assert!(parse("select named(n: lir::Add) { choose {} }").is_err());
    assert!(parse("select() { choose {} }").is_err());
    assert!(parse("select(n: lir::Add) { choose {} }").is_err());
}

#[test]
fn parse_def_abi_descriptor() {
    let input = r#"
        abi X86_64SystemV {
            arch = X86_64;
            stack = { align: 16, incoming: base(RBP, 16), outgoing: slot(8, 8) };
            args = { Integer: [RDI, RSI, RDX, RCX, R8, R9] };
            returns = { Integer: [RAX, RDX] };
            preserved = { gpr: [RBX, RBP, R12, R13, R14, R15] };
            classifier = x86_64_sysv_classifier;
        }
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::Abi(abi) = &module.defs[0] else {
        panic!("expected def-abi");
    };

    assert_eq!(abi.name, "X86_64SystemV");
    assert_eq!(abi.arch, "X86_64");
    assert_eq!(abi.stack.align, Some(16));
    assert_eq!(abi.stack.incoming_base, Some(("RBP".to_string(), 16)));
    assert_eq!(
        abi.args[0].regs,
        vec!["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
    );
    assert_eq!(abi.returns[0].regs, vec!["RAX", "RDX"]);
    assert_eq!(abi.classifier.as_deref(), Some("x86_64_sysv_classifier"));
}
