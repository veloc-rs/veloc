//! Compile real OpSpec contracts and rules to Rust, then execute the result.
use std::{fs, path::PathBuf, process::Command};
use veloc_isle::rules::{Dialects, Program, Rust};
use veloc_opgen::Source;
#[path = "../compiler.rs"]
#[allow(dead_code)] // This suite only needs the shared temporary directory helper.
mod compiler;

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
            "rule r { match: before.G_ADD(x, y), emit: after.G_ADD(x, missing), }",
            "unbound value",
        ),
        (
            "rule r { match: before.G_ADD(x), emit: after.G_ADD(x, x), }",
            "source operand count",
        ),
        (
            "rule r { match: before.G_ADD(x, x), emit: after.G_ADD(x, x), }",
            "equality guard",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emit: after.G_SADDO(x, y), }",
            "result count",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emit: after.G_FADD(x, y), }",
            "type domain",
        ),
        (
            "rule r { match: before.G_ANYEXT(x), emit: after.G_COPY(x), }",
            "independent source types",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emit: after.DoesNotExist(x, y), }",
            "unknown operation",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emit: after.G_CONSTANT(x), }",
            "properties",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emit: mir.ExtendS(x), }",
            "preconditions",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emti: after.G_ADD(x, y), }",
            "unknown rule field",
        ),
        (
            "rule r { match: before.G_ADD(x, y), emit: after.G_ADD(x, y), } rule s { match: before.G_ADD(x, y), emit: after.G_ADD(y, x), }",
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
