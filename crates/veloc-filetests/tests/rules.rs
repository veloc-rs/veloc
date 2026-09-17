//! Compile real OpSpec contracts and rules to Rust, then execute the result.
use std::{fs, path::PathBuf, process::Command};
use veloc_isle::rules::{Dialects, Program, Rust};
use veloc_opgen::Source;
#[path = "../compiler.rs"]
#[allow(dead_code)] // This suite only needs the shared temporary directory helper.
mod compiler;

#[test]
fn typed_legalization_contracts_reject_invalid_rules() {
    use veloc_isle::rules::{DecisionRust, decisions};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../veloc");
    let definitions = Source::load(root.join("lir/defs/module.ops"))
        .unwrap()
        .parse()
        .unwrap();
    let source = std::fs::read_to_string(root.join("codegen/isle/x86_64/legalize.rules")).unwrap();
    let compile = |source: &str| {
        decisions(
            source,
            &definitions,
            DecisionRust {
                dialect: "lir",
                function: "decide",
                opcode: "veloc_lir::GenericOpcode",
                result: "Action",
                value_rule: "crate::passes::lowering::LegalizeAction::values",
            },
        )
    };
    let output = compile(&source).unwrap();
    assert!(output.contains("rewrite_widen_add"));
    assert!(!output.contains("recipes.widen"));
    // Nested decisions and host identifiers share the same expression compiler.
    let nested = source.replace(
        "_ => recipes.bit_count(),",
        "_ => match true { true if false => recipes.legal(), _ => recipes.bit_count(), },",
    );
    assert!(compile(&nested).unwrap().contains("match true"));
    let named_host = source
        .replace("target: &Target", "__match_value: &Target")
        .replace("target.supports", "__match_value.supports");
    let output = compile(&named_host).unwrap();
    assert!(output.contains("__match_value_ == "));
    assert!(output.contains("__match_value.supports"));

    for (from, to, diagnostic) in [
        (
            "_ => recipes.bit_count(),",
            "",
            "final unguarded _ fallback",
        ),
        (
            "_ => recipes.bit_count(),",
            "_ if false => recipes.bit_count(),",
            "final unguarded _ fallback",
        ),
        (
            "_ => recipes.bit_count(),",
            "_ => recipes.bit_count(), Type::I32 => recipes.legal(), _ => recipes.bit_count(),",
            "unreachable arm",
        ),
        (
            "_ => recipes.bit_count(),",
            "Type::I32 => recipes.legal(), Type::I32 => recipes.legal(), _ => recipes.bit_count(),",
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
        (
            "recipes: &Recipes",
            "other: &Recipes",
            "undeclared host member",
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
    let mir = Source::load(root.join("mir/defs/module.ops"))
        .unwrap()
        .parse()
        .unwrap();
    let lir = Source::load(root.join("lir/defs/module.ops"))
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
            .join("crates/veloc-filetests/fixture/module.ops"),
    )
    .unwrap()
    .parse()
    .unwrap();
    dialects.insert("fixture", &fixture).unwrap();
    dialects
}

#[test]
fn compiled_value_rules_execute_with_an_independent_host() {
    let program = Program::compile(include_str!("fixtures/values.rules"), &dialects()).unwrap();
    let code = program
        .rust(Rust {
            function: "lower",
            context: "Context",
            source: ("before", "Before"),
            target: ("after", "After"),
        })
        .unwrap();
    let temp = compiler::Temp::new("veloc-isle-rules").unwrap();
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
