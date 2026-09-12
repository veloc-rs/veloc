//! Exercise the filesystem entry point, checked models and generated artifacts.
mod common;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use veloc_opgen::Source;

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "veloc-opgen-imports-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&path).unwrap();
        let files = Self(path);
        files.write("prelude.ops", &common::BUILTINS);
        files
    }
    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.0.join(name), text).unwrap();
    }
    fn load(&self, name: &str) -> Result<Source, veloc_opgen::SourceError> {
        Source::load(self.0.join(name))
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn diamond_imports_generate_each_definition_once() {
    let files = Files::new();
    files.write(
        "shared.ops",
        "import \"prelude.ops\";\n// no final newline\ntypeset Small = I8 | I16;\ntypeset Unused = I8;",
    );
    files.write(
        "left.ops",
        "import \"prelude.ops\";\nimport \"shared.ops\";\n",
    );
    files.write("right.ops", "import \"./shared.ops\";\n");
    files.write(
        "root.ops",
        r#"// import "not-a-dependency";
import "left.ops";
import "right.ops";
struct Unary { arg: Value }
op Example<T: Small>(arg: T) -> T {
    meta: OpInfo { memory: Known([]) }, mnemonic: "example",
    storage: Unary { arg },
}
"#,
    );
    let source = files.load("root.ops").unwrap();
    let generated = source.compile().unwrap();
    assert_eq!(generated.opcodes.matches("pub const Small:").count(), 1);
    assert!(!generated.opcodes.contains("pub const Unused:"));
    for name in [
        "root.ops",
        "left.ops",
        "right.ops",
        "shared.ops",
        "prelude.ops",
    ] {
        assert!(source.dependencies().any(|path| path == files.0.join(name)));
    }
    assert!(
        !source
            .dependencies()
            .any(|path| path.ends_with("not-a-dependency"))
    );
}

#[test]
fn imported_hosts_and_helpers_need_no_capability_configuration() {
    let files = Files::new();
    files.write(
        "host.ops",
        r#"
extern interface Numbers { fn next(n: u32) -> u32; }
fn Next(n: u32) -> u32 { value: Numbers.next(n) }
"#,
    );
    files.write(
        "helpers.ops",
        r#"
import "host.ops";
fn Twice(n: u32) -> u32 { value: Next(Next(n)) }
"#,
    );
    let consumer = r#"
import "prelude.ops";
import "helpers.ops";
interface Summary { count: u32 }
struct Data { n: u32 }
op Example(n: u32) -> () {
    meta: OpInfo { memory: Known([]) }, mnemonic: "example",
    storage: Data { n: n }, implements: [Summary { count: Twice(n) }],
}
"#;
    files.write("consumer.ops", consumer);
    let source = files.load("consumer.ops").unwrap();
    let generated = source.compile().unwrap();
    assert!(generated.host.contains("pub trait Numbers"));
    assert_eq!(
        generated
            .instructions
            .matches("host::traits::Numbers::next")
            .count(),
        2
    );
    assert!(source.dependencies().any(|path| path.ends_with("host.ops")));

    // Importing definitions is still required; there is no implicit host registry.
    files.write(
        "helpers.ops",
        "fn Twice(n: u32) -> u32 { value: Numbers.next(n) }",
    );
    let error = files.load("consumer.ops").unwrap().compile().err().unwrap();
    assert_eq!(error.path, files.0.join("helpers.ops"));
}

#[test]
fn model_errors_retain_imported_file_line_and_column() {
    let files = Files::new();
    files.write("bad.ops", "// 类型定义\ntypeset Broken = Missing;");
    files.write("root.ops", "import \"prelude.ops\";\nimport \"bad.ops\";");
    let source = files.load("root.ops").unwrap();
    let error = source.parse().err().unwrap();
    assert_eq!(error.path, files.0.join("bad.ops"));
    assert_eq!(error.diagnostic.line, 2);
    assert!(error.diagnostic.column > 1);
    assert!(error.diagnostic.message.contains("Missing"));
}

#[test]
fn imported_files_are_syntactically_independent() {
    let files = Files::new();
    files.write("bad.ops", "typeset Broken =\n");
    files.write("root.ops", "import \"bad.ops\";\n}");
    let error = files.load("root.ops").err().unwrap();
    // The root is also malformed. Neither file may complete the other's braces.
    assert!(error.diagnostic.message.contains("expected"));
    files.write("root.ops", "import \"bad.ops\";\n");
    let error = files.load("root.ops").err().unwrap();
    assert_eq!(error.path, files.0.join("bad.ops"));
    assert_eq!(error.diagnostic.line, 2);
    assert!(error.diagnostic.message.contains("imported from"));
}

#[test]
fn cycles_missing_files_and_late_imports_have_diagnostics() {
    let files = Files::new();
    files.write("a.ops", "import \"b.ops\";");
    files.write("b.ops", "import \"./a.ops\";");
    let error = files.load("a.ops").err().unwrap();
    assert!(error.to_string().contains("import cycle"));
    assert!(error.to_string().contains("b.ops"));
    files.write("missing.ops", "\nimport \"missing-target.ops\";");
    let error = files.load("missing.ops").err().unwrap();
    assert!(error.to_string().contains("missing-target.ops"));
    assert!(error.to_string().contains("missing.ops:2:1"));
    files.write("late.ops", "typeset A = I8;\nimport \"prelude.ops\";");
    assert!(
        files
            .load("late.ops")
            .err()
            .unwrap()
            .diagnostic
            .message
            .contains("precede")
    );
    assert!(
        veloc_opgen::parse("import \"prelude.ops\";")
            .err()
            .unwrap()
            .message
            .contains("Source::load")
    );
}

#[test]
fn import_strings_are_not_a_second_ad_hoc_lexer() {
    let files = Files::new();
    files.write("space name.ops", "import \"prelude.ops\";");
    files.write("root.ops", "import \"space name.ops\";\n");
    files.load("root.ops").unwrap().compile().unwrap();
    for text in [
        "import prelude;",
        "import \"\";",
        "import \"/absolute.ops\";",
        "import \"prelude.ops\"",
    ] {
        files.write("invalid.ops", text);
        assert!(files.load("invalid.ops").is_err(), "{text}");
    }
}

#[cfg(unix)]
#[test]
fn symlink_identity_and_dependencies_are_both_preserved() {
    let files = Files::new();
    std::os::unix::fs::symlink(Path::new("prelude.ops"), files.0.join("alias.ops")).unwrap();
    files.write("root.ops", "import \"prelude.ops\";\nimport \"alias.ops\";");
    let source = files.load("root.ops").unwrap();
    source.compile().unwrap();
    assert!(
        source
            .dependencies()
            .any(|path| path.ends_with("alias.ops"))
    );
    assert!(
        source
            .dependencies()
            .any(|path| path.ends_with("prelude.ops"))
    );
}

#[test]
fn production_entry_points_generate_the_same_runtime_contracts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mir = Source::load(root.join("mir/defs/module.ops")).unwrap();
    let lir = Source::load(root.join("lir/defs/module.ops")).unwrap();
    assert!(mir.compile().unwrap().opcodes.contains("pub enum Opcode"));
    assert!(
        lir.compile()
            .unwrap()
            .instructions
            .contains("pub enum GenericOpcode")
    );
    assert!(mir.dependencies().any(|p| p.ends_with("defs/types.ops")));
}

#[test]
fn original_offsets_survive_unicode_imports_and_comments() {
    let files = Files::new();
    files.write("类型.ops", "");
    files.write(
        "root.ops",
        "import \"prelude.ops\";\nimport \"类型.ops\"; // 原文保留\n\ntypeset Broken = Missing;",
    );
    let error = files.load("root.ops").unwrap().parse().err().unwrap();
    assert_eq!(error.path, files.0.join("root.ops"));
    assert_eq!(error.diagnostic.line, 4);
    assert_eq!(error.diagnostic.column, 18);
    assert!(error.diagnostic.message.contains("Missing"));
}

#[test]
fn output_plan_errors_keep_the_imported_source_location() {
    let files = Files::new();
    files.write("bad.ops", "import \"prelude.ops\"; struct Work {}\nop Work() -> () { meta: OpInfo { memory: Known([]) }, mnemonic: \"emit\", storage: Work {} }");
    files.write("root.ops", "import \"prelude.ops\";\nimport \"bad.ops\";");
    let source = files.load("root.ops").unwrap();
    source.parse().unwrap();
    let error = source.plan().err().expect("invalid output plan");
    assert_eq!(error.path, files.0.join("bad.ops"));
    assert_eq!(error.diagnostic.line, 2);
    assert!(error.diagnostic.message.contains("InstBuilder method"));
}

#[test]
fn rust_type_bindings_follow_imports_and_preserve_diagnostics() {
    let files = Files::new();
    files.write("types.ops", "type Token = rust(\"crate::tokens::Token\");");
    files.write(
        "consumer.ops",
        r#"
import "prelude.ops";
import "types.ops";
extern interface Tokens { fn read(value: Token) -> Token; }
struct Entry { value: Token }
"#,
    );
    let generated = files.load("consumer.ops").unwrap().compile().unwrap();
    assert!(generated.host.contains("value: crate::tokens::Token"));
    assert!(
        generated
            .instructions
            .contains("pub value: crate::tokens::Token")
    );
    assert!(!generated.instructions.contains("pub struct Token"));

    files.write("types.ops", "type Token = rust(\"crate::Token; invalid\");");
    let error = files.load("consumer.ops").unwrap().compile().err().unwrap();
    assert_eq!(error.path, files.0.join("types.ops"));
    assert_eq!(error.diagnostic.line, 1);
}

#[test]
fn imports_are_file_local_even_when_siblings_are_loaded_first() {
    let files = Files::new();
    for (body, name) in [
        ("struct Holder { value: Type }", "Type"),
        ("struct Holder { value: Float }", "Float"),
        (
            "fn use_float(Float: Float) -> Float { value: Float }",
            "Float",
        ),
        (
            "fn query(value: Type) -> bool { value: value.is_scalar() }",
            "Type",
        ),
        ("typeset Small = I8;", "I8"),
        (
            "type Inputs = rust(\"crate::inst::Arguments\") { field: list(Value), }",
            "Value",
        ),
    ] {
        files.write("consumer.ops", body);
        for root in [
            "import \"prelude.ops\"; import \"consumer.ops\";",
            "import \"consumer.ops\"; import \"prelude.ops\";",
        ] {
            files.write("root.ops", root);
            let error = files
                .load("root.ops")
                .unwrap()
                .parse()
                .err()
                .expect("sibling import leaked");
            assert_eq!(error.path, files.0.join("consumer.ops"));
            assert!(
                error
                    .diagnostic
                    .message
                    .contains(&format!("`{name}` is not imported")),
                "{error}"
            );
        }
        files.write("consumer.ops", &format!("import \"prelude.ops\";\n{body}"));
        files.load("root.ops").unwrap().compile().unwrap();
    }
}

#[test]
fn imported_type_sets_do_not_expose_unimported_rust_types() {
    let files = Files::new();
    files.write("scalar-types.ops", common::TYPES);
    files.write(
        "consumer.ops",
        "import \"scalar-types.ops\"; struct Holder { value: Float }",
    );
    files.write(
        "root.ops",
        "import \"prelude.ops\"; import \"consumer.ops\";",
    );
    // Avoid duplicating the type declarations: prelude reaches this same file.
    files.write(
        "prelude.ops",
        &format!(
            "import \"scalar-types.ops\";\n{}\n{}",
            include_str!("../../defs/builtins.ops")
                .strip_prefix("import \"types.ops\";\n")
                .unwrap(),
            include_str!("../../defs/comparisons.ops"),
        ),
    );
    let error = files
        .load("root.ops")
        .unwrap()
        .parse()
        .err()
        .expect("data namespace leaked");
    assert_eq!(error.path, files.0.join("consumer.ops"));
    assert!(
        error.diagnostic.message.contains("`Float` is not imported"),
        "{error}"
    );
}
