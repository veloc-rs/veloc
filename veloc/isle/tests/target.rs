//! Production target definitions from OpSpec expansion through machine emission.
use veloc_isle::target::{Def, MatchKind, Pattern, compile, parse};

#[test]
fn production_target_contracts_generate_all_consumers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../codegen/isle/x86_64");
    let definitions = veloc_opgen::Source::load(root.join("instructions.ops")).unwrap();
    let mut files = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "isle")
        })
        .collect::<Vec<_>>();
    files.sort();
    let input = files
        .iter()
        .map(|path| std::fs::read_to_string(path).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    let output = compile(&input, "x86_64", &definitions).unwrap();
    for contract in [
        "pub enum TargetInst",
        "REGISTER_VIEWS",
        "RegisterWrite::ZeroExtend",
        "REG_RAX",
        "REG_AL",
        "REG_R15D",
        "register_constraints",
        "pub fn validate",
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
    assert_eq!(output, compile(&input, "x86_64", &definitions).unwrap());
}

#[test]
fn typed_constants_and_templates_share_rust_style_syntax() {
    use veloc_opgen::syntax::{DeclKind, Kind};
    let declarations = veloc_opgen::syntax::parse(
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
fn parse_select_rule_with_node_bind_and_covers() {
    let input = r#"
        (select-rule
          (match (Add (GPR64 $x) (GPR64 $y) @n))
          (emit (X86Add64 $x $y))
          (covers (@n))
          (cost 1))
    "#;

    let module = parse(input).expect("parse should succeed");
    assert_eq!(module.defs.len(), 1);

    let Def::SelectRule(rule) = &module.defs[0] else {
        panic!("expected select-rule");
    };

    assert_eq!(rule.attrs.covers, vec!["n"]);
    assert_eq!(rule.attrs.cost, Some(1));
    assert_eq!(rule.patterns.len(), 1);
    assert!(matches!(rule.patterns[0], Pattern::NodeBind { .. }));
}

#[test]
fn parse_combine_rule_with_match_pair() {
    let input = r#"
        (combine-rule
          (match-pair ((Sdiv $lhs $rhs @q)
                       (Srem $lhs $rhs @r)))
          (when ((same_block @q @r)))
          (replace (Sdivrem $lhs $rhs))
          (covers (@q @r))
          (cost 1))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::CombineRule(rule) = &module.defs[0] else {
        panic!("expected combine-rule");
    };

    assert_eq!(rule.match_kind, MatchKind::Pair);
    assert_eq!(rule.patterns.len(), 2);
    assert_eq!(rule.attrs.covers, vec!["q", "r"]);
}

#[test]
fn parse_rewrite_rule_definition() {
    let input = r#"
        (rewrite-rule
          (match (Add (GPR64 $x) (GPR64 $y) @n))
          (replace (Add (GPR64 $y) (GPR64 $x)))
          (cost 1)
          (priority 10))
    "#;

    let module = parse(input).expect("parse should succeed");
    let Def::RewriteRule(rule) = &module.defs[0] else {
        panic!("expected rewrite-rule");
    };

    assert_eq!(rule.attrs.cost, Some(1));
    assert_eq!(rule.attrs.priority, Some(10));
    assert_eq!(rule.patterns.len(), 1);
}

#[test]
fn parse_def_abi_descriptor() {
    let input = r#"
        (def-abi X86_64SystemV
          (arch X86_64)
          (stack
            (align 16)
            (incoming-base RBP 16)
            (outgoing-slot 8 8))
          (args
            (class Integer (regs RDI RSI RDX RCX R8 R9)))
          (returns
            (class Integer (regs RAX RDX)))
          (preserved
            (gpr RBX RBP R12 R13 R14 R15))
          (classifier x86_64_sysv_classifier))
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
