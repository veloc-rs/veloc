//! Generate Rust, compile against explicit hosts, execute, and check diagnostics.
use super::common;
use std::{fs, path::Path, process::Command};

fn checked(predicate: &str) -> Result<veloc_opgen::Generated, veloc_opgen::Error> {
    common::compile(&format!(
        r#"
fn Double(n: u64) -> u64 {{ value: n * 2 }}
struct Custom {{ bits: u64, yes: bool }}
op Example(number: u64, flag: bool) -> ScalarInteger {{
    meta: OpInfo {{ memory: MemoryEffect::NONE }},
    mnemonic: "example", storage: Custom {{ bits: number, yes: flag }},
    verify {{
        {predicate};
    }}
}}
"#
    ))
}

#[test]
fn generated_rust_executes_checked_arithmetic_and_short_circuit_loops() {
    // Compile the actual emitted Rust, not a second interpreter for the AST.
    // Small host adapters make accidental eager pool reads observable as panics.
    let mut code = String::from(
        r#"
#![allow(dead_code)]
extern crate self as veloc_types;
type ModuleData = ();
type Inst = usize;
type Type = ();
type Result<T> = std::result::Result<T, String>;
mod inst { #[derive(Clone, Copy)] pub struct ConstantPoolId(pub usize); }
mod dfg {
    impl crate::inst::ConstantPoolId {
        pub fn get(self, data: &[Vec<u8>]) -> Option<&Vec<u8>> {
            assert_ne!(self.0, 99, "unreachable property was read");
            data.get(self.0)
        }
    }
}
"#,
    );
    for (index, (predicate, expected)) in [
        ("number * number > 0", [true, true, false, false]),
        ("flag || number * number > 0", [true, true, false, true]),
        ("number * 2 + 1 == 7", [true, true, false, false]),
        ("-number < 0", [true, true, true, true]),
        ("number - 2 - 1 == 0", [true, true, false, false]),
        (
            "!(number <= 3) || flag && number != 0",
            [false, true, true, true],
        ),
        ("(number + 1) * 2 >= 8", [true, true, true, true]),
        ("Double(number) > 0", [true, true, false, false]),
        ("flag || Double(number + 0) > 0", [true, true, false, true]),
    ]
    .iter()
    .enumerate()
    {
        let validation = checked(predicate).unwrap().validation;
        code.push_str(&format!(r#"
mod numeric_{index} {{
    use super::*;
    enum Opcode {{ Example }}
    enum ViewData {{ Custom {{ bits: u64, yes: bool }} }}
    type InstView<'a> = ViewData;
    impl ViewData {{ fn opcode(&self) -> Opcode {{ Opcode::Example }} }}
    struct Function;
    impl Function {{ fn constraint_error(&self, _: Inst, message: &str) -> String {{ message.into() }} }}
    {validation}
    #[test] fn execute() {{
        let f = Function;
        for ((bits, yes), expected) in [(3, false), (3, true), (u64::MAX, false), (u64::MAX, true)].into_iter().zip({expected:?}) {{
            assert_eq!(f.validate_constraints(&(), 0, &ViewData::Custom {{ bits, yes }}, &[], &[]).is_ok(), expected);
        }}
    }}
}}
"#));
    }
    for (index, (predicate, valid)) in [
        ("all(data, |i| i != 0 && len(other) > 0)", false),
        ("all(data, |i| all(data, |i| i < 8) && i < 8)", true),
        ("all(data, |i| true)", true),
        ("all(data, |i| Above([2, 3], i))", true),
    ]
    .iter()
    .enumerate()
    {
        let validation = common::compile(&format!(
            r#"
fn Above(items: array(u32, 2), limit: i128) -> bool {{
    value: all(items, |item| i128(item) > limit)
}}
struct Buffers {{ first: ConstantPoolId, second: ConstantPoolId }}
op Example(data: Bytes, other: Bytes) -> Vector {{
    meta: OpInfo {{ memory: MemoryEffect::NONE }},
    mnemonic: "example", storage: Buffers {{ first: pool(data), second: pool(other) }},
    text: "{{data:bytes}}, {{other:bytes}}",
    verify {{
        {predicate};
    }}
}}
"#
        ))
        .unwrap()
        .validation;
        code.push_str(&format!(r#"
mod sequences_{index} {{
    use super::*;
    enum Opcode {{ Example }}
    type InstView<'a> = ViewData;
    enum ViewData {{ Buffers {{ first: inst::ConstantPoolId, second: inst::ConstantPoolId }} }}
    impl ViewData {{ fn opcode(&self) -> Opcode {{ Opcode::Example }} }}
    struct Function {{ dfg: Vec<Vec<u8>> }}
    impl Function {{ fn constraint_error(&self, _: Inst, message: &str) -> String {{ message.into() }} }}
    {validation}
    #[test] fn execute() {{
        let data = ViewData::Buffers {{ first: inst::ConstantPoolId(0), second: inst::ConstantPoolId(99) }};
        let f = Function {{ dfg: vec![vec![0, 1]] }};
        assert_eq!(f.validate_constraints(&(), 0, &data, &[], &[]).is_ok(), {valid});
        let empty = Function {{ dfg: vec![vec![]] }};
        assert!(empty.validate_constraints(&(), 0, &data, &[], &[]).is_ok());
        let missing = Function {{ dfg: vec![] }};
        assert!(missing.validate_constraints(&(), 0, &data, &[], &[]).is_err());
    }}
}}
"#));
    }
    let generated = common::compile(
        r#"
extern interface Arithmetic {
    fn next(n: u64) -> optional(u64);
}
fn Next(n: u64) -> u64 { value: Arithmetic::next(n)? }
fn Successor(n: u64) -> u64 { value: Next(n) }
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Type::I32 {
    meta: OpInfo { memory: MemoryEffect::NONE }, mnemonic: "example",
    storage: Custom { bits: number, yes: flag },
    verify { require(flag || Successor(number) > number, "host failure"); }
}
"#,
    )
    .unwrap();
    let host = generated.host;
    let validation = generated.validation;
    code.push_str(&format!(r#"
type VectorConst = ();
type FuncId = u32;
type SigId = u32;
mod host {{
    pub mod traits {{ {host} }}
    pub struct Context;
    impl Context {{
        pub fn new(_: &()) -> Self {{ Self }}
        pub fn with_module(self, _: &(), _: ()) -> Self {{ self }}
    }}
    impl traits::Arithmetic for Context {{
        fn next(&self, n: u64) -> Option<u64> {{ n.checked_add(1) }}
    }}
}}
mod host_calls {{
    use super::*;
    enum Opcode {{ Example }}
    enum ViewData {{ Custom {{ bits: u64, yes: bool }} }}
    type InstView<'a> = ViewData;
    impl ViewData {{ fn opcode(&self) -> Opcode {{ Opcode::Example }} }}
    struct Function {{ dfg: (), signature: () }}
    impl Function {{ fn constraint_error(&self, _: Inst, message: &str) -> String {{ message.into() }} }}
    {validation}
    #[test] fn execute() {{
        let f = Function {{ dfg: (), signature: () }};
        for (bits, yes, valid) in [(3, false, true), (u64::MAX, false, false), (u64::MAX, true, true)] {{
            let result = f.validate_constraints(&(), 0, &ViewData::Custom {{ bits, yes }}, &[], &[]);
            assert_eq!(result.is_ok(), valid);
            if !valid {{ assert_eq!(result.unwrap_err(), "host failure"); }}
        }}
    }}
}}
"#));
    let dir = common::compiler::Temp::new("veloc-generated").unwrap();
    let input = dir.join("generated.rs");
    let binary = dir.join(format!("generated{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&input, &code).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(&rustc)
        .args(["--edition=2024", "--test", "-Dwarnings"])
        .arg(&input)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = std::process::Command::new(binary).output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // The same generated calls must fail without the trait implementation,
    // even when an identically named inherent method is available.
    std::fs::write(
        &input,
        code.replace("impl traits::Arithmetic for Context", "impl Context"),
    )
    .unwrap();
    let output = std::process::Command::new(&rustc)
        .args(["--edition=2024", "--test", "--emit=metadata"])
        .arg(&input)
        .arg("-o")
        .arg(dir.join("missing-host.rmeta"))
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "missing host implementation was accepted"
    );
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostic.contains("Arithmetic") && diagnostic.contains("not satisfied"),
        "{diagnostic}"
    );
}

#[test]
fn generated_mapping_executes_in_logical_argument_order() {
    let generated = common::compile(
        r#"
storage Operands { prefix: "G_" }
struct Pair { right: Use, high: Def, left: Use, low: Def }
op G_PAIR(first: Type::I32, second: Type::I64) -> (low: Type::I32, high: Type::I64) {
    meta: OpInfo { memory: MemoryEffect::NONE },
    storage: Pair { right: second, high, left: first, low },
}
"#,
    )
    .unwrap();
    // Compile the actual builder emitted by opgen. The small machine container
    // observes encoded operand order without duplicating the projection logic.
    let start = generated
        .instructions
        .find("impl MachineInst { pub fn build_pair")
        .unwrap();
    let mut depth = 0;
    let mut end = start;
    for (offset, ch) in generated.instructions[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = start + offset + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    let builder = &generated.instructions[start..end];
    let code = format!(
        r#"
#![allow(dead_code, non_camel_case_types)]
type Reg = u32;
#[derive(Debug, PartialEq)]
struct Writable<T>(T);
#[derive(Debug, PartialEq)]
enum MachineOperand {{ Def(Writable<Reg>), Use(Reg) }}
enum GenericOpcode {{ G_PAIR }}
enum MachineOpcode {{ Generic(GenericOpcode) }}
struct MachineInst {{ operands: Vec<MachineOperand> }}
impl MachineInst {{
    fn build_generic(_: MachineOpcode, operands: Vec<MachineOperand>) -> Self {{ Self {{ operands }} }}
}}
mod smallvec {{
    macro_rules! smallvec {{ ($($operand:expr),* $(,)?) => {{ vec![$($operand),*] }}; }}
    pub(crate) use smallvec;
}}
{builder}
fn main() {{
    let inst = MachineInst::build_pair(Writable(10), Writable(20), 30, 40);
    assert_eq!(inst.operands, vec![
        MachineOperand::Use(40), MachineOperand::Def(Writable(20)),
        MachineOperand::Use(30), MachineOperand::Def(Writable(10)),
    ]);
}}
"#
    );
    let dir = common::compiler::Temp::new("veloc-generated").unwrap();
    let input = dir.join("generated.rs");
    let binary = dir.join(format!("generated{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&input, code).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(rustc)
        .args(["--edition=2024", "-Dwarnings"])
        .arg(&input)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = std::process::Command::new(binary).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

struct Fixture(common::compiler::Temp);

impl Fixture {
    fn new() -> Self {
        Self(common::compiler::Temp::new("opgen-rust").unwrap())
    }

    fn compile(&self, source: &str) -> std::process::Output {
        let input = self.0.join("main.rs");
        fs::write(&input, source).unwrap();
        Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
            .args(["--edition=2024", "--crate-name=fixture"])
            .arg(input)
            .arg("-o")
            .arg(self.0.join("run"))
            .output()
            .unwrap()
    }
}

#[test]
fn rust_evaluates_foreign_const_metadata_without_a_query_catalog() {
    let fixture = Fixture::new();
    let defs = r#"
type Token = rust("crate::Token") {
    const BASE: Self;
    const fn from_number(value: u32) -> Self;
    const fn number(self) -> optional(u32);
    fn runtime(self) -> u32;
}
const fn token(n: u32) -> Token = rust("crate::token");
const fn doubled(n: u32) -> u32 { value: n * 2 }
const fn positive(items: array(u32, 2)) -> bool { value: all(items, |n| n > 0) }
struct Info { count: u32, wide: i128, ok: bool, next: optional(u32) }
struct Empty {}
op Check() -> () {
    meta: Info {
        count: doubled(Token::from_number(Token::BASE.number()?).number()?) + 1,
        wide: i128(token(7).number()?) + 10,
        ok: positive([token(7).number()?, 8])
            && all(token(7).number(), |n| n > 0)
            && (token(7).number()? > 0 || token(0).number()? > 0),
        next: token(7).number(),
    },
    mnemonic: "check", storage: Empty {},
}
"#;
    let generated = veloc_opgen::compile(defs).unwrap();
    assert!(
        generated
            .opcodes
            .contains("as crate::type_methods::TokenConst>::number")
    );
    let declarations = veloc_opgen::syntax::parse(defs).unwrap();
    let traits =
        veloc_opgen::interfaces::declarations(&declarations, defs, "crate::type_methods").unwrap();
    let code = format!(
        r#"
#![feature(const_trait_impl)]
#![allow(dead_code, unused_variables, unreachable_code, unused_parens)]
#[derive(Clone, Copy)] pub struct Token(u32);
pub const fn token(n: u32) -> Token {{ Token(n) }}
pub mod type_methods {{ {traits} }}
const impl type_methods::TokenConst for Token {{
    const BASE: Self = Self(7);
    fn from_number(value: u32) -> Self {{ Self(value) }}
    fn number(self) -> Option<u32> {{ if self.0 == 0 {{ None }} else {{ Some(self.0) }} }}
}}
impl type_methods::Token for Token {{ fn runtime(self) -> u32 {{ self.0 }} }}
#[derive(Clone, Copy)] pub struct Type;
impl Type {{
    fn element(self) -> Option<u8> {{ None }}
    fn lane_count(&self) -> u16 {{ 1 }}
    fn is_scalable(&self) -> bool {{ false }}
}}
mod inst {{
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Info {{ pub count: u32, pub wide: i128, pub ok: bool, pub next: Option<u32> }}
    {}
}}
fn main() {{
    let info = inst::Opcode::Check.meta();
    assert_eq!(info.count, 15);
    assert_eq!(info.wide, 17);
    assert!(info.ok);
    assert_eq!(info.next, Some(7));
}}
"#,
        generated.opcodes
    );
    let result = fixture.compile(&code);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        Command::new(fixture.0.join("run"))
            .status()
            .unwrap()
            .success()
    );

    // A regular impl is insufficient, even if its signature is identical.
    let result = fixture.compile(&code.replace(
        "const impl type_methods::TokenConst",
        "impl type_methods::TokenConst",
    ));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("const"));

    // The generator cannot evaluate this method; rustc catches absent results.
    let result = fixture.compile(&code.replace("crate::token(7u32)", "crate::token(0u32)"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("invalid constant expression"));

    let result = fixture.compile(&code.replace("Some(self.0)", "Some(u32::MAX)"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("invalid constant expression"));
}

#[test]
fn generated_traits_require_an_explicit_rust_implementation() {
    let dir = std::env::temp_dir().join(format!(
        "opgen-interface-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    fs::create_dir(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    let defs = dir.join("interface.ops");
    fs::write(
        &defs,
        r#"
type Token = rust("fixture::Token") {
    trait: rust("fixture::traits::Token"),
    fn count(self) -> u32;
}
"#,
    )
    .unwrap();
    let source = veloc_opgen::Source::load(&defs).unwrap();
    let traits = source.interfaces("fixture::traits").unwrap();
    assert!(!traits.contains("impl "));
    let input = dir.join("main.rs");
    let binary = dir.join(format!("run{}", std::env::consts::EXE_SUFFIX));
    let code = format!(
        r#"
extern crate self as fixture;
pub struct Token;
pub mod traits {{ {traits} }}
impl traits::Token for Token {{ fn count(self) -> u32 {{ 7 }} }}
fn main() {{ assert_eq!(traits::Token::count(Token), 7); }}
"#
    );
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let compile = || {
        Command::new(&rustc)
            .args(["--edition=2024", "-Dwarnings"])
            .arg(&input)
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap()
    };
    fs::write(&input, &code).unwrap();
    let result = compile();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(Command::new(&binary).status().unwrap().success());
    fs::write(&input, code.replace("fn count(self) -> u32 { 7 }", "")).unwrap();
    let result = compile();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("E0046"));
}

#[test]
fn a_rust_path_does_not_implicitly_supply_a_type_catalog() {
    let error = veloc_opgen::parse(
        r#"
type Type = rust("veloc_types::Type");
typeset Small = Type::I32;
"#,
    )
    .err()
    .expect("foreign paths do not declare logical types");
    assert!(error.message.contains("unknown type or typeset"), "{error}");
}

#[test]
fn generated_files_are_formatted_together_and_invalid_syntax_is_reported() {
    let dir = common::compiler::Temp::new("opgen-format").unwrap();
    let files = [dir.join("functions.rs"), dir.join("types.rs")];
    // Generated files can declare modules without owning their source files.
    fs::write(
        &files[0],
        "mod external;\nfn choose(x:bool)->u32{match x{true=>1,false=>2}}\n",
    )
    .unwrap();
    fs::write(&files[1], "struct Example{value:u32}\n").unwrap();
    let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../rustfmt.toml");
    veloc_opgen::format_rust(&files, &config).unwrap();
    let formatted = files
        .each_ref()
        .map(|path| fs::read_to_string(path).unwrap());
    assert!(formatted[0].contains("    match x {\n        true => 1,"));
    assert!(formatted[1].contains("struct Example {\n    value: u32,\n}"));
    veloc_opgen::format_rust(&files, &config).unwrap();
    assert_eq!(
        formatted,
        files
            .each_ref()
            .map(|path| fs::read_to_string(path).unwrap())
    );

    fs::write(&files[0], "fn broken( {").unwrap();
    let error = veloc_opgen::format_rust(&files, &config).unwrap_err();
    assert!(error.to_string().contains("rustfmt failed"));
}
