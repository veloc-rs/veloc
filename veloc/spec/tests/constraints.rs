mod common;

fn checked(predicate: &str) -> Result<veloc_opgen::Generated, veloc_opgen::Error> {
    common::compile(&format!(
        r#"
fn Double(n: u64) -> u64 {{ value: n * 2 }}
struct Custom {{ bits: u64, yes: bool }}
op Example(number: u64, flag: bool) -> ScalarInteger {{
    meta: OpInfo {{ memory: Known([]) }},
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
    meta: OpInfo {{ memory: Known([]) }},
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
fn Next(n: u64) -> u64 { value: Arithmetic.next(n)? }
fn Successor(n: u64) -> u64 { value: Next(n) }
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> I32 {
    meta: OpInfo { memory: Known([]) }, mnemonic: "example",
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
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("veloc-constraints-{}-{unique}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
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
