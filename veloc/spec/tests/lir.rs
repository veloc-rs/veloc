mod common;

const BINARY: &str =
    "storage Operands { prefix: \"G_\" }\nstruct Binary { dst: Def, lhs: Use, rhs: Use }";
const ADD: &str = "op G_SUM<T: Integer>(lhs: T, rhs: T) -> (dst: T) { meta: OpInfo {}, storage: Binary { dst, lhs, rhs }, semantics: bv.add(lhs, rhs) }";

#[test]
fn invalid_definitions_fail_before_emission() {
    for (source, message) in [
        (
            format!(
                "{BINARY} {}",
                ADD.replace("storage: Binary", "storage: Missing")
            ),
            "unknown operand format",
        ),
        (
            format!("{BINARY} {}", ADD.replace("rhs: T", "other: T")),
            "unknown input",
        ),
        (
            format!(
                "{BINARY} {}",
                ADD.replace("-> (dst: T)", "-> (dst: T, extra: T)")
            ),
            "has no storage mapping",
        ),
        (
            format!("{BINARY} {}", ADD.replace("Integer", "Missing")),
            "Missing",
        ),
        (
            format!("{BINARY} {}", ADD.replace("rhs)", "missing)")),
            "unknown semantic value",
        ),
        (
            format!("{BINARY} {}", ADD.replace("Integer", "Float")),
            "floating-point",
        ),
        (
            format!(
                "{BINARY} {}",
                ADD.replace("semantics:", "flow: Call, semantics:")
            ),
            "MAY_TRAP",
        ),
        (format!("{BINARY} {ADD} {ADD}"), "duplicate op"),
        (
            "storage Operands {} struct Bad { dst: Def, dst: Use }".into(),
            "duplicate field",
        ),
        (
            "storage Operands {} struct Bad { values: Uses, dst: Def }".into(),
            "entire operand sequence",
        ),
        (
            "storage Operands {} struct Bad { dst: Def } layout Bad { lengths: [0] }".into(),
            "layout overrides",
        ),
        (
            "storage Operands {} struct Bad { dst: Def } layout Bad { lengths: [1, 1] }".into(),
            "layout overrides",
        ),
        (
            format!(
                "{BINARY} {}",
                ADD.replace("semantics:", "arity: 3, semantics:")
            ),
            "arity",
        ),
    ] {
        let error = veloc_opgen::parse(&common::source(&source))
            .err()
            .expect(&source);
        assert!(error.message.contains(message), "{source}\n{error}");
    }
}

#[test]
fn unsupported_output_contracts_are_not_silently_ignored() {
    let source = common::source(&format!(
        "{BINARY} {}",
        ADD.replace("semantics:", "text: \"{lhs}, {rhs}\", semantics:")
    ));
    assert!(
        veloc_opgen::compile(&source)
            .err()
            .unwrap()
            .message
            .contains("does not yet support")
    );
}

#[test]
fn generated_mapping_executes_in_logical_argument_order() {
    let generated = common::compile(
        r#"
storage Operands { prefix: "G_" }
struct Pair { right: Use, high: Def, left: Use, low: Def }
op G_PAIR(first: Type.I32, second: Type.I64) -> (low: Type.I32, high: Type.I64) {
    meta: OpInfo { memory: Known([]) },
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
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("veloc-storage-{}-{unique}", std::process::id()));
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
