//! Compile real OpSpec contracts and rules to Rust, then execute the result.
use std::{fs, path::PathBuf, process::Command};
use veloc_spec::Source;
use veloc_spec::rules::{Dialects, Program, Rust};
#[path = "../compiler.rs"]
#[allow(dead_code)] // This suite only needs the shared temporary directory helper.
mod compiler;

#[test]
fn selector_types_share_domains_and_reject_unknown_types() {
    use veloc_spec::{Emit, Options, Target};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../veloc");
    let definitions = Source::load(root.join("codegen/defs/x86_64/instructions.spec")).unwrap();
    let lir = Source::load(root.join("lir/defs/module.spec")).unwrap();
    let header = [
        "types/defs/types.spec",
        "defs/type_sets.spec",
        "codegen/defs/x86_64/predicates.spec",
        "codegen/defs/x86_64/cpu/features.spec",
    ]
    .map(|path| {
        fs::read_to_string(root.join(path))
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("import "))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .join("\n");
    let rule = r#"
typeset Small = Type::I8 | Type::I16 | Type::I32;
select(n: lir::Constant) {
    choose {
        case {
            require(type_is<Small>(n.dst));
            replace(n, build(X86Mov32Imm(n.imm)));
        }
    }
}
"#;
    let compile = |text: &str| {
        compiler::source(&format!("{header}\n{text}"))
            .unwrap()
            .generate(
                &[Emit::Selector],
                Options {
                    target: Some(Target {
                        input: Some(("lir", &lir)),
                        arch: "x86_64",
                        context: "crate::Host",
                        definitions: &definitions,
                    }),
                    ..Default::default()
                },
            )
    };
    let output = compile(rule).unwrap();
    let code = output.get(Emit::Selector).unwrap();
    assert!(code.contains("ctx.get_type(reg)"));
    assert!(code.contains("veloc_types::Type::I32"));
    assert!(!code.contains("is_i32"));
    for (from, to) in [
        ("Type::I8", "Type::MISSING"),
        ("type_is<Small>(n.dst)", "type_is<Missing>(n.dst)"),
        ("type_is<Small>(n.dst)", "type_is<Small>(n.dst, other)"),
    ] {
        assert!(compile(&rule.replace(from, to)).is_err());
    }
}

#[test]
fn typed_legalization_contracts_reject_invalid_rules() {
    use veloc_spec::rules::{DecisionRust, decisions};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../veloc");
    let definitions = Source::load(root.join("lir/defs/module.spec"))
        .unwrap()
        .parse()
        .unwrap();
    let strip_imports = |text: String| {
        text.lines()
            .filter(|line| !line.starts_with("import "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let header = ["types/defs/types.spec", "defs/type_sets.spec"]
        .map(|path| strip_imports(std::fs::read_to_string(root.join(path)).unwrap()))
        .join("\n");
    let target = format!(
        "{header}\n{}",
        strip_imports(
            std::fs::read_to_string(root.join("codegen/defs/x86_64/legalize.spec")).unwrap()
        )
    );
    let shared =
        strip_imports(std::fs::read_to_string(root.join("codegen/defs/legalize.spec")).unwrap());
    let mut source = format!("{target}\n{shared}");
    // Exercise nested host decisions independently of target policy spelling.
    source.push_str(
        r#"
rule decision_test<T: Word>(inst: lir::Ctpop<T>, target: &Target) {
    action = match T {
        Type::I32 if target.supports(Instruction::POPCNT32) => legal,
        _ => legal,
    };
}
"#,
    );
    let compile = |source: &str| {
        decisions(
            source,
            &definitions,
            DecisionRust {
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
        )
    };
    // Moving templates before or after policy cannot affect decision priority.
    let decision = |code: String| {
        code.split("pub fn decide")
            .nth(1)
            .unwrap()
            .split("_ => None,\n} }\n")
            .next()
            .unwrap()
            .to_owned()
    };
    assert_eq!(
        decision(compile(&format!("{target}\n{shared}")).unwrap()),
        decision(compile(&format!("{shared}\n{target}")).unwrap())
    );
    // Templates alone are checked and generated, but create no matching cases.
    let declarations = target.split("rule ret_0").next().unwrap();
    let templates_only = compile(&format!("{declarations}\n{shared}")).unwrap();
    assert!(!templates_only.contains("GenericOpcode::Ctpop =>"));
    assert!(!templates_only.contains("GenericOpcode::Ctlz =>"));
    assert!(!templates_only.contains("GenericOpcode::Cttz =>"));
    for (from, to, message) in [
        (
            "IntCC::LtS",
            "FloatCC::Lt",
            "attribute constant type mismatch",
        ),
        ("IntCC::LtS", "IntCC::Missing", "undeclared constant"),
    ] {
        let error = compile(&source.replace(from, to)).unwrap_err();
        assert!(error.message.contains(message), "{}", error.message);
    }
    let renamed = compile(
        &source
            .replace("fn emit(&mut self", "fn construct(&mut self")
            .replace("emit = emit;", "emit = construct;"),
    )
    .unwrap();
    assert!(renamed.contains("ctx.construct("));
    for (from, to, message) in [
        (
            "emit = emit;",
            "emit = missing;",
            "undeclared rewrite method",
        ),
        (
            "result: optional(RewriteValue)",
            "result: usize",
            "invalid signature for rewrite role emit",
        ),
    ] {
        let error = compile(&source.replace(from, to)).unwrap_err();
        assert!(error.message.contains(message), "{}", error.message);
    }
    let output = compile(&source).unwrap();
    assert!(!output.contains("ctx.input("));
    assert!(!output.contains("ctx.value_type("));
    assert!(!output.contains("ctx.bind("));
    assert!(output.contains("destination: Option<veloc_lir::Reg>"));
    assert!(!output.contains("pub trait Query"));
    assert!(output.contains("&impl crate::passes::lowering::legalize::contracts::Query"));
    // Rules use the shared Type representation without a second constant trait.
    assert!(output.contains("veloc_types::Type::PTR"));
    assert!(!output.contains("pub trait Type {"));
    assert!(
        compile(&source.replace("Type::I32 if", "Type::MISSING if"))
            .unwrap_err()
            .message
            .contains("undeclared constant")
    );
    assert!(output.contains("rewrite_widen_add"));
    assert!(!output.contains("Recipes"));
    assert!(output.contains("fn rewrite_load_displacement_host()"));
    let unused = source.replace("expand(load_displacement, inst)", "legal");
    assert!(
        compile(&unused)
            .unwrap()
            .contains("fn rewrite_load_displacement_host()")
    );
    // Nested decisions and host identifiers share the same expression compiler.
    let nested = source.replace(
        "_ => legal,",
        "_ => match true { true if false => legal, _ => legal, },",
    );
    assert!(compile(&nested).unwrap().contains("match true"));
    let named_host = source
        .replace("target: &Target", "__match_value: &Target")
        .replace("target.supports", "__match_value.supports");
    let output = compile(&named_host).unwrap();
    assert!(output.contains("__match_value_ == "));
    assert!(output.contains("__match_value.supports"));

    // Fragments compose without introducing root mutations or runtime calls.
    let composed = format!(
        "{source}\n{}",
        r#"
fn twice<T: Word>(x: T) -> T {
    lir::Add<T>(x, x)
}
fn nested<U: Word>(x: U) -> U {
    let a = twice<U>(x);
    twice<U>(a)
}
rewrite composition(inst: lir::Ctpop<Type::I32>) {
    replace = nested<Type::I32>(lir::Constant<Type::I32>(9));
}
"#
    );
    let generated = compile(&composed).unwrap();
    let body = generated
        .split("fn rewrite_composition_case1")
        .nth(1)
        .unwrap();
    assert_eq!(body.matches("GenericOpcode::Constant").count(), 1);
    assert_eq!(body.matches("GenericOpcode::Add").count(), 2);
    assert!(!body.contains("nested("));
    for (extra, diagnostic) in [
        ("fn bad<T: Word>(x: T) -> T { bad<T>(x) }", "recursive"),
        (
            "fn bad<T: Word>(x: T) -> T { other<T>(x) } fn other<U: Word>(x: U) -> U { bad<U>(x) }",
            "recursive",
        ),
        (
            "fn bad(x: Type::I32) -> Type::I64 { x }",
            "result type mismatch",
        ),
        (
            "fn bad(x: Type::I32) -> Type::I32 { low_bit<Type::I64>(x) }",
            "argument type mismatch",
        ),
        (
            "fn bad(x: Type::F32) -> Type::F32 { low_bit<Type::F32>(x) }",
            "outside domain",
        ),
        ("fn bad(x: Type::I32) -> Type::I32 { low_bit(x) }", "arity"),
        ("fn bad<T: Word>(x: T) -> T { inst.src }", "unbound value"),
        (
            "fn bad(x: Type::I32) -> Type::I32 { let x = x; x }",
            "duplicate",
        ),
    ] {
        let error = compile(&format!("{source}\n{extra}")).unwrap_err();
        assert!(error.message.contains(diagnostic), "{error}");
    }

    for (from, to, diagnostic) in [
        (
            "expand(load_displacement, inst)",
            "expand(store_displacement, inst)",
            "node signature",
        ),
        (
            "lowering::legalize::displacement",
            "lowering::legalize::displacement;panic!()",
            "Rust",
        ),
        (
            "expand(popcount32, inst)",
            "expand(missing, inst)",
            "unknown rewrite",
        ),
        (
            "expand(popcount32, inst)",
            "expand(popcount64, inst)",
            "type domain",
        ),
        (
            "expand(popcount32, inst)",
            "expand(leading_zeros32, inst)",
            "node signature",
        ),
        (
            "expand(popcount32, inst)",
            "expand(popcount32, unknown)",
            "matched instruction",
        ),
        ("rewrite popcount32", "rule popcount32", "unknown rewrite"),
        (
            "let nibbles =",
            "let pairs =",
            "duplicate replacement binding",
        ),
        (
            "lir::Lshr<Type::I32>(pairs,",
            "lir::Lshr<Type::I32>(missing,",
            "unbound value",
        ),
        (
            "lir::Constant<Type::I32>(1)",
            "lir::Constant<Type::F32>(1)",
            "does not accept",
        ),
        (
            "lir::Constant<Type::I32>(1)",
            "lir::Constant<Type::I32>(9223372036854775808)",
            "exceeds i64",
        ),
        ("_ => legal,", "", "final unguarded _ fallback"),
        (
            "_ => legal,",
            "_ if false => legal,",
            "final unguarded _ fallback",
        ),
        (
            "_ => legal,",
            "_ => legal, Type::I32 => legal, _ => legal,",
            "unreachable arm",
        ),
        (
            "_ => legal,",
            "Type::I32 => legal, Type::I32 => legal, _ => legal,",
            "unreachable repeated",
        ),
        (
            "Type::I32 if target.supports",
            "unknown if target.supports",
            "match patterns",
        ),
        (
            "Type::I32 if target.supports",
            "Type::UNDECLARED if target.supports",
            "undeclared constant",
        ),
        (
            "inst: lir::Add<T>",
            "inst: unknown::Add<T>",
            "unknown instruction namespace",
        ),
        (
            "inst: lir::Add<T>",
            "inst: lir::Missing<T>",
            "unknown operation",
        ),
        (
            "inst: lir::Add<T>",
            "inst: lir::Add<T, T>",
            "type arguments",
        ),
        (
            "lir::Zext<Type::I32>(inst.lhs)",
            "lir::Zext<Type::F32>(inst.lhs)",
            "does not accept",
        ),
        (
            "replace = lir::Trunc<T>(lir::Add",
            "replace = lir::Trunc<Type::I8>(lir::Add",
            "replacement result",
        ),
        (
            "lir::Zext<Type::I32>(inst.rhs)",
            "lir::Zext<Type::I32>(inst.missing)",
            "unbound value",
        ),
        ("<T: Narrow>(inst", "<T: Unknown>(inst", "unknown type set"),
        (
            "lir::Add<Type::I32>(lir::Zext",
            "lir::Add<Type::I64>(lir::Zext",
            "requires equal",
        ),
        (
            "target: &Target",
            "other: &Target",
            "undeclared host member",
        ),
        (
            "target.supports(",
            "target.undeclared(",
            "undeclared host member",
        ),
        (
            "Instruction::POPCNT32",
            "Instruction::UNKNOWN",
            "undeclared constant",
        ),
        (
            "replace = lir::Trunc<T>",
            "emit = lir::Trunc<T>",
            "unknown decision rule field",
        ),
    ] {
        assert!(source.contains(from));
        let error = compile(&source.replacen(from, to, 1)).unwrap_err();
        assert!(
            error.message.contains(diagnostic),
            "expected {diagnostic}, got {error}"
        );
    }
}

fn dialects() -> Dialects {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../veloc");
    let mir = Source::load(root.join("mir/defs/module.spec"))
        .unwrap()
        .parse()
        .unwrap();
    let lir = Source::load(root.join("lir/defs/module.spec"))
        .unwrap()
        .parse()
        .unwrap();
    let mut dialects = Dialects::default();
    dialects.insert("mir", &mir).unwrap();
    dialects.insert("before", &lir).unwrap();
    dialects.insert("after", &lir).unwrap();
    let fixture = Source::load(
        root.parent()
            .unwrap()
            .join("crates/veloc-filetests/fixture/module.spec"),
    )
    .unwrap()
    .parse()
    .unwrap();
    dialects.insert("fixture", &fixture).unwrap();
    dialects
}

#[test]
fn compiled_value_adapters_execute_with_an_independent_host() {
    let program = Program::compile(include_str!("fixtures/values.spec"), &dialects()).unwrap();
    let code = program
        .rust(Rust {
            function: "lower",
            context: "Context",
            source: ("before", "Before"),
            target: ("after", "After"),
        })
        .unwrap();
    let temp = compiler::Temp::new("veloc-spec-rules").unwrap();
    let source = temp.join("rules.rs");
    let executable = temp.join(format!("rules{}", std::env::consts::EXE_SUFFIX));
    fs::write(
        &source,
        format!("{}\n{code}", include_str!("fixtures/values.rs")),
    )
    .unwrap();
    let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "-o"])
        .arg(&executable)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(executable).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn invalid_rules_fail_before_rust_generation() {
    let dialects = dialects();
    for (source, message) in [
        (
            "rule r { match = before.Add(x, y); emit = after.Add(x, missing); }",
            "unbound value",
        ),
        (
            "rule r { match = before.Add(x); emit = after.Add(x, x); }",
            "source operand count",
        ),
        (
            "rule r { match = before.Add(x, x); emit = after.Add(x, x); }",
            "equality guard",
        ),
        (
            "rule r { match = before.Add(x, y); emit = after.Saddo(x, y); }",
            "result count",
        ),
        (
            "rule r { match = before.Add(x, y); emit = after.Fadd(x, y); }",
            "type domain",
        ),
        (
            "rule r { match = before.Anyext(x); emit = after.Copy(x); }",
            "independent source types",
        ),
        (
            "rule r { match = before.Add(x, y); emit = after.DoesNotExist(x, y); }",
            "unknown operation",
        ),
        (
            "rule r { match = before.Add(x, y); emit = after.Constant(x); }",
            "properties",
        ),
        (
            "rule r { match = before.Add(x, y); emit = mir.ExtendS(x); }",
            "preconditions",
        ),
        (
            "rule r { match = before.Add(x, y); emti = after.Add(x, y); }",
            "unknown rule field",
        ),
        (
            "rule r { match = before.Add(x, y); emit = after.Add(x, y); } rule s { match = before.Add(x, y); emit = after.Add(y, x); }",
            "overlapping",
        ),
    ] {
        let error = Program::compile(source, &dialects)
            .err()
            .expect("invalid rule accepted");
        assert!(error.message.contains(message), "{source}: {error}");
        assert!(error.line > 0 && error.column > 0);
    }
}

#[test]
fn primitive_inference_uses_the_same_contract_checker() {
    let dialects = dialects();
    let mut program = Program::compile("", &dialects).unwrap();
    program.infer_primitives(&dialects, "mir", "after").unwrap();
    let roots: Vec<_> = program.roots().collect();
    assert!(roots.contains(&"mir.IAdd"));
    assert!(roots.contains(&"mir.IAnd"));
    assert!(!roots.contains(&"mir.IDivS"));
    assert!(!roots.contains(&"mir.INeg"));
    program
        .infer_primitives(&dialects, "fixture", "after")
        .unwrap();
    let roots: Vec<_> = program.roots().collect();
    assert!(roots.contains(&"fixture.Direct"));
    for name in [
        "fixture.Reversed",
        "fixture.Composed",
        "fixture.Trapping",
        "fixture.Multiple",
    ] {
        assert!(
            !roots.contains(&name),
            "{name} is not a direct pure primitive"
        );
    }
    let mut ambiguous = Program::compile("", &dialects).unwrap();
    assert!(
        ambiguous
            .infer_primitives(&dialects, "fixture", "fixture")
            .unwrap_err()
            .message
            .contains("ambiguous")
    );
}

#[test]
fn construction_functions_compose_with_checked_rust_bindings() {
    use veloc_spec::rules::{DecisionRust, decisions};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../veloc");
    let definitions = Source::load(root.join("lir/defs/module.spec"))
        .unwrap()
        .parse()
        .unwrap();
    let code = decisions(
        include_str!("fixtures/construction.spec"),
        &definitions,
        DecisionRust {
            dialect: "lir",
            function: "decide",
            opcode: "crate::Opcode",
            field: "crate::Field",
            result: "Action",
            value_interface: "ValueRules",
            value_adapter: "crate::replace_values",
            rewrite: "crate::rewrite",
            legal_action: "crate::unused_legal",
        },
    )
    .unwrap();
    let temp = compiler::Temp::new("veloc-construction").unwrap();
    let source = temp.join("construction.rs");
    let executable = temp.join(format!("construction{}", std::env::consts::EXE_SUFFIX));
    let host = include_str!("fixtures/construction.rs");
    for (host, valid) in [
        (host.to_owned(), true),
        (
            host.replace(
                "ctx.emit(Opcode::Add, ty, &[x, x], &[], None)",
                "ctx.editor()",
            ),
            false,
        ),
        (host.replace("-> Value {", "-> () {"), false),
    ] {
        fs::write(&source, format!("{host}\nmod generated {{ {code} }}")).unwrap();
        let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
            .args(["--edition=2024", "-o"])
            .arg(&executable)
            .arg(&source)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            valid,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if valid {
            let output = Command::new(&executable).output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
