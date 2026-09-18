//! Exercise the filesystem entry point, checked models and generated artifacts.
use super::common;

use std::path::Path;
use veloc_spec::Source;

struct Files(common::compiler::Temp);
impl Files {
    fn new() -> Self {
        let files = Self(common::compiler::Temp::new("veloc-spec-imports").unwrap());
        files.write("prelude.spec", &common::BUILTINS);
        files
    }
    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.0.join(name), text).unwrap();
    }
    fn load(&self, name: &str) -> Result<Source, veloc_spec::SourceError> {
        common::load(self.0.join(name))
    }
}

#[test]
fn diamond_imports_generate_each_definition_once() {
    let files = Files::new();
    files.write(
        "shared.spec",
        "import \"prelude.spec\";\n// no final newline\ntypeset Small = Type::I8 | Type::I16;\ntypeset Unused = Type::I8;",
    );
    files.write(
        "left.spec",
        "import \"prelude.spec\";\nimport \"shared.spec\";\n",
    );
    files.write("right.spec", "import \"./shared.spec\";\n");
    files.write(
        "root.spec",
        r#"// import "not-a-dependency";
import "left.spec";
import "right.spec";
struct Unary { arg: Value }
op Example<T: Small>(arg: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "example";
    storage = Unary { arg };
}
"#,
    );
    let source = files.load("root.spec").unwrap();
    let generated = source.compile().unwrap();
    assert_eq!(
        generated[veloc_spec::Emit::Opcodes]
            .matches("pub const Small:")
            .count(),
        1
    );
    assert!(!generated[veloc_spec::Emit::Opcodes].contains("pub const Unused:"));
    // Importing shared contracts uses the Rust definitions, not parallel types.
    for declaration in [
        "pub struct MemFlags",
        "pub struct OpTraits",
        "pub struct MemoryEffects",
        "pub enum MemoryEffect",
    ] {
        assert!(!generated[veloc_spec::Emit::Opcodes].contains(declaration));
        assert!(!generated[veloc_spec::Emit::Instructions].contains(declaration));
    }
    for name in [
        "root.spec",
        "left.spec",
        "right.spec",
        "shared.spec",
        "prelude.spec",
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
        "host.spec",
        r#"
type Numbers = rust("crate::host::Numbers") { fn next(&self, n: u32) -> u32; }
fn Next(ctx: &Numbers, n: u32) -> u32 { value = ctx.next(n); }
"#,
    );
    files.write(
        "helpers.spec",
        r#"
import "host.spec";
fn Twice(ctx: &Numbers, n: u32) -> u32 { value = Next(ctx, Next(ctx, n)); }
"#,
    );
    let consumer = r#"
import "prelude.spec";
import "helpers.spec";
import "host.spec";
struct Summary { count: u32 }
struct Data { n: u32 }
op Example(n: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "example";
    storage = Data { n: n }; query summary(ctx: Numbers) -> Summary { count: Twice(ctx, n) }
}
"#;
    files.write("consumer.spec", consumer);
    let source = files.load("consumer.spec").unwrap();
    let generated = source.compile().unwrap();
    assert!(generated[veloc_spec::Emit::Instructions].contains("pub trait Numbers"));
    assert_eq!(
        generated[veloc_spec::Emit::Instructions]
            .matches("crate::type_methods::Numbers>::next")
            .count(),
        2
    );
    assert!(
        source
            .dependencies()
            .any(|path| path.ends_with("host.spec"))
    );

    // Importing definitions is still required; there is no implicit host registry.
    files.write(
        "helpers.spec",
        "fn Twice(ctx: &Numbers, n: u32) -> u32 { value = ctx.next(n); }",
    );
    let error = files
        .load("consumer.spec")
        .unwrap()
        .compile()
        .err()
        .unwrap();
    assert_eq!(error.path, files.0.join("helpers.spec"));
}

#[test]
fn model_errors_retain_imported_file_line_and_column() {
    let files = Files::new();
    files.write("bad.spec", "// 类型定义\ntypeset Broken = Missing;");
    files.write(
        "root.spec",
        "import \"prelude.spec\";\nimport \"bad.spec\";",
    );
    let source = files.load("root.spec").unwrap();
    let error = source.parse().err().unwrap();
    assert_eq!(error.path, files.0.join("bad.spec"));
    assert_eq!(error.diagnostic.line, 2);
    assert!(error.diagnostic.column > 1);
    assert!(error.diagnostic.message.contains("Missing"));
}

#[test]
fn imported_files_are_syntactically_independent() {
    let files = Files::new();
    files.write("bad.spec", "typeset Broken = \n");
    files.write("root.spec", "import \"bad.spec\";\n}");
    let error = files.load("root.spec").err().unwrap();
    // The root is also malformed. Neither file may complete the other's braces.
    assert!(error.diagnostic.message.contains("expected"));
    files.write("root.spec", "import \"bad.spec\";\n");
    let error = files.load("root.spec").err().unwrap();
    assert_eq!(error.path, files.0.join("bad.spec"));
    assert_eq!(error.diagnostic.line, 2);
    assert!(error.diagnostic.message.contains("imported from"));
}

#[test]
fn cycles_missing_files_and_late_imports_have_diagnostics() {
    let files = Files::new();
    files.write("a.spec", "import \"b.spec\";");
    files.write("b.spec", "import \"./a.spec\";");
    let error = files.load("a.spec").err().unwrap();
    assert!(error.to_string().contains("import cycle"));
    assert!(error.to_string().contains("b.spec"));
    files.write("missing.spec", "\nimport \"missing-target.spec\";");
    let error = files.load("missing.spec").err().unwrap();
    assert!(error.to_string().contains("missing-target.spec"));
    assert!(error.to_string().contains("missing.spec:2:1"));
    files.write(
        "late.spec",
        "typeset A = Type::I8;\nimport \"prelude.spec\";",
    );
    assert!(
        files
            .load("late.spec")
            .err()
            .unwrap()
            .diagnostic
            .message
            .contains("precede")
    );
}

#[test]
fn import_strings_are_not_a_second_ad_hoc_lexer() {
    let files = Files::new();
    files.write("space name.spec", "import \"prelude.spec\";");
    files.write("root.spec", "import \"space name.spec\";\n");
    files.load("root.spec").unwrap().compile().unwrap();
    for text in [
        "import prelude;",
        "import \"\";",
        "import \"/absolute.spec\";",
        "import \"prelude.spec\"",
    ] {
        files.write("invalid.spec", text);
        assert!(files.load("invalid.spec").is_err(), "{text}");
    }
}

#[cfg(unix)]
#[test]
fn symlink_identity_and_dependencies_are_both_preserved() {
    let files = Files::new();
    std::os::unix::fs::symlink(Path::new("prelude.spec"), files.0.join("alias.spec")).unwrap();
    files.write(
        "root.spec",
        "import \"prelude.spec\";\nimport \"alias.spec\";",
    );
    let source = files.load("root.spec").unwrap();
    source.compile().unwrap();
    assert!(
        source
            .dependencies()
            .any(|path| path.ends_with("alias.spec"))
    );
    assert!(
        source
            .dependencies()
            .any(|path| path.ends_with("prelude.spec"))
    );
}

#[test]
fn production_entry_points_generate_the_same_runtime_contracts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../veloc");
    let mir = common::load(root.join("mir/defs/module.spec")).unwrap();
    let lir = common::load(root.join("lir/defs/module.spec")).unwrap();
    assert!(mir.compile().unwrap()[veloc_spec::Emit::Opcodes].contains("pub enum Opcode"));
    assert!(
        lir.compile().unwrap()[veloc_spec::Emit::Instructions].contains("pub enum GenericOpcode")
    );
    assert!(mir.dependencies().any(|p| p.ends_with("defs/types.spec")));
}

#[test]
fn original_offsets_survive_unicode_imports_and_comments() {
    let files = Files::new();
    files.write("类型.spec", "");
    files.write(
        "root.spec",
        "import \"prelude.spec\";\nimport \"类型.spec\"; // 原文保留\n\ntypeset Broken = Missing;",
    );
    let error = files.load("root.spec").unwrap().parse().err().unwrap();
    assert_eq!(error.path, files.0.join("root.spec"));
    assert_eq!(error.diagnostic.line, 4);
    assert_eq!(error.diagnostic.column, 18);
    assert!(error.diagnostic.message.contains("Missing"));
}

#[test]
fn output_plan_errors_keep_the_imported_source_location() {
    let files = Files::new();
    files.write("bad.spec", "import \"prelude.spec\"; struct Work {}\nop Work() -> () { meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = \"emit\"; storage = Work {}; }");
    files.write(
        "root.spec",
        "import \"prelude.spec\";\nimport \"bad.spec\";",
    );
    let source = files.load("root.spec").unwrap();
    source.parse().unwrap();
    let error = source.plan().err().expect("invalid output plan");
    assert_eq!(error.path, files.0.join("bad.spec"));
    assert_eq!(error.diagnostic.line, 2);
    assert!(error.diagnostic.message.contains("InstBuilder method"));
}

#[test]
fn rust_type_bindings_follow_imports_and_preserve_diagnostics() {
    let files = Files::new();
    files.write("types.spec", "type Token = rust(\"crate::tokens::Token\");");
    files.write(
        "consumer.spec",
        r#"
import "prelude.spec";
import "types.spec";
type Tokens = rust("crate::host::Tokens") { fn read(&self, value: Token) -> Token; }
struct Entry { value: Token }
"#,
    );
    let generated = files.load("consumer.spec").unwrap().compile().unwrap();
    assert!(generated[veloc_spec::Emit::Instructions].contains("value: crate::tokens::Token"));
    assert!(generated[veloc_spec::Emit::Instructions].contains("pub value: crate::tokens::Token"));
    assert!(!generated[veloc_spec::Emit::Instructions].contains("pub struct Token"));

    files.write(
        "types.spec",
        "type Token = rust(\"crate::Token; invalid\");",
    );
    let error = files
        .load("consumer.spec")
        .unwrap()
        .compile()
        .err()
        .unwrap();
    assert_eq!(error.path, files.0.join("types.spec"));
    assert_eq!(error.diagnostic.line, 1);
}

#[test]
fn imports_are_file_local_even_when_siblings_are_loaded_first() {
    let files = Files::new();
    for (body, name) in [
        ("struct Holder { value: Type }", "Type"),
        ("struct Holder { value: Float }", "Float"),
        ("struct Holder { value: MemFlags }", "MemFlags"),
        ("struct Holder { value: OpTraits }", "OpTraits"),
        ("struct Holder { value: MemoryEffects }", "MemoryEffects"),
        ("struct Holder { value: MemoryEffect }", "MemoryEffect"),
        (
            "fn use_float(Float: Float) -> Float { value = Float; }",
            "Float",
        ),
        (
            "fn query(value: Type) -> bool { value = value.is_scalar(); }",
            "Type",
        ),
        ("typeset Small = Type::I8;", "Type"),
        ("type SIMD = Type::I32X4;", "Type"),
        (
            "type Inputs = rust(\"crate::inst::Arguments\") { field = list(Value); }",
            "Value",
        ),
    ] {
        files.write("consumer.spec", body);
        for root in [
            "import \"prelude.spec\"; import \"consumer.spec\";",
            "import \"consumer.spec\"; import \"prelude.spec\";",
        ] {
            files.write("root.spec", root);
            let error = files
                .load("root.spec")
                .unwrap()
                .parse()
                .err()
                .expect("sibling import leaked");
            assert_eq!(error.path, files.0.join("consumer.spec"));
            assert!(
                error
                    .diagnostic
                    .message
                    .contains(&format!("`{name}` is not imported")),
                "{error}"
            );
        }
        files.write(
            "consumer.spec",
            &format!("import \"prelude.spec\";\n{body}"),
        );
        files.load("root.spec").unwrap().compile().unwrap();
    }

    files.write(
        "references.spec",
        r#"type Ref = rust("crate::Value") { field = operand; }"#,
    );
    let consumer = r#"
import "prelude.spec";
struct Inputs { args: ValueList }
op Consume(args: sequence(Ref)) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "consume"; storage = Inputs { args };
}
"#;
    for consumer in [
        consumer.to_owned(),
        consumer
            .replace("args: ValueList", "args: Value")
            .replace("sequence(Ref)", "Ref<Type::I32>"),
    ] {
        files.write("consumer.spec", &consumer);
        files.write(
            "root.spec",
            r#"import "references.spec"; import "consumer.spec";"#,
        );
        let error = files
            .load("root.spec")
            .unwrap()
            .parse()
            .err()
            .expect("SSA reference import leaked");
        assert_eq!(error.path, files.0.join("consumer.spec"));
        assert!(
            error.diagnostic.message.contains("`Ref` is not imported"),
            "{error}"
        );
        files.write(
            "consumer.spec",
            &format!("import \"references.spec\";\n{consumer}"),
        );
        files.load("root.spec").unwrap().compile().unwrap();
    }
}

#[test]
fn imported_type_sets_do_not_expose_unimported_rust_types() {
    let files = Files::new();
    files.write(
        "scalar-types.spec",
        "type F32 = float(32); type F64 = float(64); type Type = rust(\"crate::Type\"); typeset Float = Type::F32 | Type::F64;",
    );
    files.write(
        "consumer.spec",
        "import \"scalar-types.spec\"; struct Holder { value: Float }",
    );
    files.write(
        "root.spec",
        "import \"prelude.spec\"; import \"consumer.spec\";",
    );
    files.write(
        "prelude.spec",
        "import \"scalar-types.spec\"; type Float = rust(\"crate::Float\");",
    );
    let error = files
        .load("root.spec")
        .unwrap()
        .parse()
        .err()
        .expect("data namespace leaked");
    assert_eq!(error.path, files.0.join("consumer.spec"));
    assert!(
        error.diagnostic.message.contains("`Float` is not imported"),
        "{error}"
    );
}
