mod common;

fn checked(predicate: &str) -> Result<veloc_opgen::Generated, veloc_opgen::Error> {
    common::compile_mir(&format!(
        r#"
format Custom {{ fields: [opcode(Opcode), bits(u64), yes(bool)], opcode: dynamic(opcode) }}
op Example(@number: u64, @flag: bool) -> ScalarInteger {{
    mnemonic: "example", storage: Custom {{ bits: number, yes: flag }},
    memory: NONE, constraints: [{predicate}]
}}
"#
    ))
}

#[test]
fn constant_folding_preserves_precedence_and_short_circuiting() {
    for predicate in [
        "1 + 2 * 3 == 7",
        "(1 + 2) * 3 == 9",
        "-3 + 2 == -1",
        "true || number > 0",
        "!(false && flag)",
    ] {
        let code = checked(predicate).unwrap().validation;
        assert!(!code.contains("constraint_error"), "{predicate}: {code}");
        assert!(!code.contains("InstructionData::Custom"));
    }
    assert!(
        checked("1 + 2 * 3 == 9")
            .err()
            .unwrap()
            .message
            .contains("always false")
    );
}

#[test]
fn record_access_uses_logical_names_not_rule_or_storage_names() {
    let source = [
        include_str!("../../mir/defs/formats.ops"),
        include_str!("../../mir/defs/mir.ops"),
    ]
    .join("\n");
    let renamed = source
        .replace("@imm: PtrIndexImm", "@stride: PtrIndexImm")
        .replace("pool(imm)", "pool(stride)")
        .replace("imm.scale", "stride.scale")
        .replace("imm.offset", "stride.offset");
    assert!(
        common::compile_mir(&renamed)
            .unwrap()
            .validation
            .contains(".scale")
    );
    assert!(common::compile_mir(&source.replace("imm.scale != 0", "imm.unknown != 0")).is_err());
}

#[test]
fn generated_rust_executes_checked_arithmetic_and_short_circuit_loops() {
    // Compile the actual emitted Rust, not a second interpreter for the AST.
    // Small host adapters make accidental eager pool reads observable as panics.
    let mut code = String::from(
        r#"
#![allow(dead_code)]
type Inst = usize;
type Type = ();
type Result<T> = std::result::Result<T, String>;
mod inst { #[derive(Clone, Copy)] pub struct ConstantPoolId(pub usize); }
mod dfg {
    pub trait PoolKey { fn get(self, data: &[Vec<u8>]) -> Option<&Vec<u8>>; }
    impl PoolKey for crate::inst::ConstantPoolId {
        fn get(self, data: &[Vec<u8>]) -> Option<&Vec<u8>> {
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
    ]
    .iter()
    .enumerate()
    {
        let validation = checked(predicate).unwrap().validation;
        code.push_str(&format!(r#"
mod numeric_{index} {{
    use super::*;
    enum Opcode {{ Example }}
    enum InstructionData {{ Custom {{ opcode: Opcode, bits: u64, yes: bool }} }}
    impl InstructionData {{ fn opcode(&self) -> Opcode {{ Opcode::Example }} }}
    struct Function;
    impl Function {{ fn constraint_error(&self, _: Inst, message: &str) -> String {{ message.into() }} }}
    {validation}
    #[test] fn execute() {{
        let f = Function;
        for ((bits, yes), expected) in [(3, false), (3, true), (u64::MAX, false), (u64::MAX, true)].into_iter().zip({expected:?}) {{
            assert_eq!(f.validate_constraints(0, &InstructionData::Custom {{ opcode: Opcode::Example, bits, yes }}, &[], &[]).is_ok(), expected);
        }}
    }}
}}
"#));
    }
    for (index, (predicate, valid)) in [
        ("all(data, |i| i != 0 && len(other) > 0)", false),
        ("all(data, |i| all(data, |i| i < 8) && i < 8)", true),
        ("all(data, |i| true)", true),
    ]
    .iter()
    .enumerate()
    {
        let validation = common::compile_mir(&format!(r#"
format Buffers {{ fields: [opcode(Opcode), first(ConstantPoolId), second(ConstantPoolId)], opcode: dynamic(opcode) }}
op Example(@data: Bytes, @other: Bytes) -> Vector {{
    mnemonic: "example", storage: Buffers {{ first: pool(data), second: pool(other) }},
    text: Text {{ args: [bytes(data), bytes(other)] }}, memory: NONE,
    constraints: [{predicate}]
}}
"#)).unwrap().validation;
        code.push_str(&format!(r#"
mod sequences_{index} {{
    use super::*;
    enum Opcode {{ Example }}
    enum InstructionData {{ Buffers {{ opcode: Opcode, first: inst::ConstantPoolId, second: inst::ConstantPoolId }} }}
    impl InstructionData {{ fn opcode(&self) -> Opcode {{ Opcode::Example }} }}
    struct Function {{ dfg: Vec<Vec<u8>> }}
    impl Function {{ fn constraint_error(&self, _: Inst, message: &str) -> String {{ message.into() }} }}
    {validation}
    #[test] fn execute() {{
        let data = InstructionData::Buffers {{ opcode: Opcode::Example, first: inst::ConstantPoolId(0), second: inst::ConstantPoolId(99) }};
        let f = Function {{ dfg: vec![vec![0, 1]] }};
        assert_eq!(f.validate_constraints(0, &data, &[], &[]).is_ok(), {valid});
        let empty = Function {{ dfg: vec![vec![]] }};
        assert!(empty.validate_constraints(0, &data, &[], &[]).is_ok());
        let missing = Function {{ dfg: vec![] }};
        assert!(missing.validate_constraints(0, &data, &[], &[]).is_err());
    }}
}}
"#));
    }
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
    std::fs::write(&input, code).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(rustc)
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
}
