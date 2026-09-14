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
    pub type DataFlowGraph = Vec<Vec<u8>>;
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
            assert_eq!(f.validate_constraints(&Vec::new(), &(), 0, &ViewData::Custom {{ bits, yes }}, &[], &[]).is_ok(), expected);
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
        ("all(data, data, |a, b| a == b)", true),
        ("all(data, data, data, |a, b, c| a == b && b == c)", true),
        (
            "all(data, suffix(data, len(data)), |a, b| len(other) > 0)",
            false,
        ),
        ("all(data, data, |a, b| a != 0 && len(other) > 0)", false),
        (
            "all(suffix(data, len(data)), suffix(data, len(data)), |a, b| len(other) > 0)",
            true,
        ),
        (
            "all(data, data, |a, b| all(data, |a| a < 8) && a == b)",
            true,
        ),
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
    struct Function;
    impl Function {{ fn constraint_error(&self, _: Inst, message: &str) -> String {{ message.into() }} }}
    {validation}
    #[test] fn execute() {{
        let data = ViewData::Buffers {{ first: inst::ConstantPoolId(0), second: inst::ConstantPoolId(99) }};
        let f = Function;
        let buffers = vec![vec![0, 1]];
        assert_eq!(f.validate_constraints(&buffers, &(), 0, &data, &[], &[]).is_ok(), {valid});
        let empty = vec![vec![]];
        assert!(f.validate_constraints(&empty, &(), 0, &data, &[], &[]).is_ok());
        let missing = vec![];
        assert!(f.validate_constraints(&missing, &(), 0, &data, &[], &[]).is_err());
    }}
}}
"#));
    }
    let ops = r#"
type Arithmetic = rust("crate::host::Arithmetic") {
    fn snapshot(&self) -> &Snapshot;
}
type Snapshot = rust("crate::host::Snapshot") {
    fn next(&self, n: u64) -> optional(u64);
    fn limit(&self) -> u64;
}
fn Next(ctx: &Snapshot, n: u64) -> u64 { value: ctx.next(n)? }
fn Successor(ctx: &Snapshot, n: u64) -> u64 { value: Next(ctx, n) }
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Type::I32 {
    meta: OpInfo { memory: MemoryEffect::NONE }, mnemonic: "example",
    storage: Custom { bits: number, yes: flag },
    verify(ctx: Arithmetic) {
        let snapshot = ctx.snapshot();
        require(flag || Successor(snapshot, number) > number, "host failure");
        require(snapshot.limit() >= number, "limit");
    }
}
"#;
    let generated = common::compile(ops).unwrap();
    let host = veloc_opgen::interfaces::declarations(
        &veloc_opgen::syntax::parse(ops).unwrap(),
        ops,
        "crate::type_methods",
    )
    .unwrap();
    let validation = generated.validation;
    code.push_str(&format!(r#"
type VectorConst = ();
type FuncId = u32;
type SigId = u32;
pub mod type_methods {{ {host} }}
mod host {{
    pub type Arithmetic = Context;
    pub type Snapshot = Context;
    pub struct Context {{ reads: std::cell::Cell<u32> }}
    impl Context {{
        pub fn new() -> Self {{ Self {{ reads: std::cell::Cell::new(0) }} }}
    }}
    impl crate::type_methods::Arithmetic for Context {{
        fn snapshot(&self) -> &Snapshot {{
            assert_eq!(self.reads.replace(1), 0, "binding must be evaluated once");
            self
        }}
    }}
    impl crate::type_methods::Snapshot for Context {{
        fn next(&self, n: u64) -> Option<u64> {{ n.checked_add(1) }}
        fn limit(&self) -> u64 {{ u64::MAX }}
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
            let context = host::Context::new();
            let result = f.validate_constraints(&Vec::new(), &(), 0, &ViewData::Custom {{ bits, yes }}, &[], &[], &context);
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
        code.replace(
            "impl crate::type_methods::Arithmetic for Context",
            "impl Context",
        ),
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
    // A returned view must borrow the receiver, not a temporary context.
    let escaped = code.replace(
        "            self\n        }",
        "            &Self { reads: std::cell::Cell::new(0) }\n        }",
    );
    assert_ne!(escaped, code);
    std::fs::write(&input, escaped).unwrap();
    let output = std::process::Command::new(&rustc)
        .args(["--edition=2024", "--test", "--emit=metadata"])
        .arg(&input)
        .arg("-o")
        .arg(dir.join("escaped-view.rmeta"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("E0515"));
}

#[test]
fn generated_mapping_executes_in_logical_argument_order() {
    let generated = common::compile(
        r#"
type Cell = rust("crate::Cell");
struct Tag { bits: u32 }
struct Summary { bits: u32, ty: Type }
type Limits = rust("crate::Limits") {
    trait: rust("crate::LimitsInfo"),
    fn max(&self) -> u32;
}
enum Payload { variants: [Tag(Tag), Number(i64)] }
storage Operands { opcode: Code, view: View, reader: Read, writer: Build, register: Cell, attributes: Payload }
struct Pair { tag: optional(Tag), right: Cell, high: Cell, left: Cell, low: Cell }
op Pair(move first: Type::I32, second: Type::I64, tag: Tag) -> (low: Type::I32, high: Type::I64) {
    meta: OpInfo { memory: MemoryEffect::NONE },
    storage: Pair { tag: some(tag), right: second, high, left: first, low },
    text: "{first}, {second}, tag={tag.bits}",
    query summary -> Summary { bits: tag.bits, ty: first.ty() }
    verify(ctx: Limits) {
        require(tag.bits != 0, "zero tag");
        require(tag.bits <= ctx.max(), "tag exceeds limit");
    }
}
"#,
    )
    .unwrap();
    // Compile the actual builder emitted by opgen. The small machine container
    // observes encoded operand order without duplicating the projection logic.
    let start = generated.instructions.find("pub trait Build").unwrap();
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
    let views = generated
        .instructions
        .find("#[derive(Debug, Clone, Copy)] pub enum View")
        .unwrap();
    let reader = &generated.instructions[views..start];
    assert!(generated.type_rules.contains("impl crate::Code"));
    assert!(!generated.instructions.contains("GenericOpcode"));
    let signatures = common::compile(r#"
type Cell = rust("crate::Cell");
enum SigField { variants: [Sig(SigId), Number(i64)] }
storage Operands { opcode: SigCode, view: SigView, reader: SigRead, writer: SigBuild, register: Cell, attributes: SigField }
struct Call { outputs: sequence(Cell), sig: SigId, args: sequence(Cell) }
op Invoke(sig: SigId, args: sequence(Value)) -> signature {
    meta: OpInfo { memory: MemoryEffect::NONE },
    storage: Call { outputs: results(), sig, args },
    signature: sig,
}
"#).unwrap();
    let begin = signatures
        .instructions
        .find("#[derive(Debug, Clone, Copy)] pub enum SigView")
        .unwrap();
    let end = signatures.instructions.find("pub trait SigBuild").unwrap();
    let signature_reader = &signatures.instructions[begin..end];
    let signature_host = r#"
    use super::*;
    #[derive(Debug, Clone, Copy)] pub enum SigCode { Invoke }
    #[derive(Debug, Clone, Copy)] pub enum SigField { Sig(SigId), Number(i64) }
    #[derive(Clone, Copy)] struct Call<'a> { inputs: &'a [Cell], results: &'a [Cell], fields: &'a [SigField] }
    impl<'a> SigRead<'a> for Call<'a> {
        type Error = String;
        fn opcode(self) -> Option<SigCode> { Some(SigCode::Invoke) }
        fn inputs(self) -> &'a [Cell] { self.inputs }
        fn results(self) -> &'a [Cell] { self.results }
        fn fields(self) -> &'a [SigField] { self.fields }
        fn error(self, message: &str) -> String { message.into() }
        fn value_type(self, value: Cell) -> Type { value }
        fn signature(self, id: SigId) -> Option<(&'a [Type], &'a [Type])> {
            (id == 0).then_some((&[1], &[2]))
        }
    }
    pub fn check() {
        let call = Call { inputs: &[1], results: &[2], fields: &[SigField::Sig(0)] };
        call.validate().unwrap();
        assert!(Call { inputs: &[], ..call }.validate().unwrap_err().contains("value count mismatch"));
        assert!(Call { inputs: &[2], ..call }.validate().unwrap_err().contains("value 0 type mismatch"));
        assert!(Call { results: &[1], ..call }.validate().unwrap_err().contains("result 0 type mismatch"));
        assert!(Call { fields: &[SigField::Sig(1)], ..call }.validate().unwrap_err().contains("missing function or signature"));
    }
"#;
    let text_host = r#"
#[derive(Clone, Copy, PartialEq)]
struct MemFlags(bool);
impl MemFlags { fn empty() -> Self { Self(false) } }
mod atom {
    use super::*;
    use crate::text::*;
    pub trait AtomCodec {
        type Owned;
        type View<'a>: ?Sized;
        fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>, ty: Option<Type>) -> Result<Self::Owned, ParseError>;
        fn print(cx: &InstPrinter<'_>, out: &mut dyn core::fmt::Write, value: &Self::View<'_>, ty: Option<Type>) -> core::fmt::Result;
    }
    pub struct Decimal<T>(core::marker::PhantomData<T>);
    impl AtomCodec for u32 {
        type Owned = u32;
        type View<'a> = u32;
        fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>, _: Option<Type>) -> Result<u32, ParseError> {
            input.word()?.parse().map_err(|_| input.error("invalid number"))
        }
        fn print(_: &InstPrinter<'_>, out: &mut dyn core::fmt::Write, value: &u32, _: Option<Type>) -> core::fmt::Result {
            write!(out, "{value}")
        }
    }
    impl AtomCodec for Decimal<u32> {
        type Owned = u32;
        type View<'a> = u32;
        fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>, ty: Option<Type>) -> Result<u32, ParseError> {
            <u32 as AtomCodec>::parse(cx, input, ty)
        }
        fn print(cx: &InstPrinter<'_>, out: &mut dyn core::fmt::Write, value: &u32, ty: Option<Type>) -> core::fmt::Result {
            <u32 as AtomCodec>::print(cx, out, value, ty)
        }
    }
}
"#;
    let parser_host = r#"
    use super::*;
    #[derive(Debug)] pub struct ParseError(String);
    impl ParseError {
        fn context(self, context: &str) -> Self { Self(format!("{context}: {}", self.0)) }
    }
    pub struct Location;
    impl Location {
        fn error(&self, message: impl ToString) -> ParseError { ParseError(message.to_string()) }
    }
    pub struct Cursor<'a>(&'a str);
    enum Kind { Comma, Equal }
    impl<'a> Cursor<'a> {
        fn at_end(&self) -> bool { self.0.trim().is_empty() }
        fn location(&self) -> Location { Location }
        pub fn error(&self, message: impl ToString) -> ParseError { Location.error(message) }
        fn eat(&mut self, kind: Kind) -> bool {
            let c = match kind { Kind::Comma => ',', Kind::Equal => '=' };
            if let Some(rest) = self.0.trim_start().strip_prefix(c) { self.0 = rest; true } else { false }
        }
        fn expect(&mut self, kind: Kind) -> Result<(), ParseError> {
            if self.eat(kind) { Ok(()) } else { Err(self.error("missing punctuation")) }
        }
        pub fn word(&mut self) -> Result<&'a str, ParseError> {
            let input = self.0.trim_start();
            let n = input.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(input.len());
            if n == 0 { return Err(self.error("expected word")); }
            self.0 = &input[n..];
            Ok(&input[..n])
        }
    }
    pub struct OperandParser<'a>(pub &'a mut Vec<(Vec<Cell>, Vec<Cell>, Vec<Payload>)>);
    impl OperandParser<'_> {
        fn write(&mut self, op: Code, results: &[Cell], inputs: &[Cell], fields: &[Payload]) -> InstId {
            Sink { store: self.0 }.write(op, results, inputs, fields)
        }
    }
    pub struct InstPrinter<'a>(core::marker::PhantomData<&'a ()>);
    impl InstPrinter<'_> {
        fn fmt_head(&self, out: &mut dyn core::fmt::Write, name: &str, _: MemFlags) -> core::fmt::Result {
            out.write_str(name)
        }
    }
    pub fn roundtrip() {
        let mut store = Vec::new();
        let mut parser = OperandParser(&mut store);
        let id = parser.parse(Code::Pair, MemFlags::empty(), &mut Cursor("30, 40, tag=99"), None, &[10, 20]).unwrap();
        assert_eq!(id, 0);
        assert!(parser.parse(Code::Pair, MemFlags::empty(), &mut Cursor("30, 40, tag=99"), None, &[10]).is_err());
        assert!(parser.parse(Code::Pair, MemFlags::empty(), &mut Cursor("30, 40, tag=99, tag=1"), None, &[10, 20]).is_err());
        assert!(parser.parse(Code::Pair, MemFlags::empty(), &mut Cursor("30, 40"), None, &[10, 20]).is_err());
        assert!(parser.parse(Code::Pair, MemFlags(true), &mut Cursor("30, 40, tag=99"), None, &[10, 20]).is_err());
        assert_eq!(store.len(), 1);
        assert_eq!(store[id], (vec![10, 20], vec![40, 30], vec![Payload::Tag(Tag { bits: 99 })]));
        let mut text = String::new();
        InstPrinter(core::marker::PhantomData).fmt_instruction_data(&mut text, Handle(&store[id]), None).unwrap();
        let (_, operands) = text.split_once(' ').unwrap();
        let id2 = OperandParser(&mut store).parse(Code::Pair, MemFlags::empty(), &mut Cursor(operands), None, &[10, 20]).unwrap();
        assert_eq!(store[id], store[id2]);
    }
"#;
    let text_parser = &generated.text_parser;
    let text_printer = &generated.text_printer;
    let code = format!(
        r#"
#![allow(dead_code, non_camel_case_types)]
extern crate alloc;
extern crate self as veloc_types;
pub type SigId = u32;
type Cell = u32;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tag {{ pub bits: u32 }}
pub type Type = u32;
pub struct Summary {{ pub bits: u32, pub ty: Type }}
pub struct Limits(u32);
pub trait LimitsInfo {{ fn max(&self) -> u32; }}
impl LimitsInfo for Limits {{ fn max(&self) -> u32 {{ self.0 }} }}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Payload {{ Tag(Tag), Number(i64) }}
#[derive(Debug, Clone, Copy)]
pub enum Code {{ Pair }}
type InstId = usize;
struct Sink<'a> {{ store: &'a mut Vec<(Vec<Cell>, Vec<Cell>, Vec<Payload>)> }}
impl Build for Sink<'_> {{
    type Inst = usize;
    type Def = Cell;
    fn reg(value: Cell) -> Cell {{ value }}
    fn write(self, _: Code, results: &[Cell], inputs: &[Cell], fields: &[Payload]) -> InstId {{
        let id = self.store.len();
        self.store.push((results.to_vec(), inputs.to_vec(), fields.to_vec()));
        id
    }}
}}
{builder}
{reader}
mod signatures {{
{signature_host}
{signature_reader}
}}
{text_host}
mod text {{
{parser_host}
{text_parser}
{text_printer}
}}
#[derive(Clone, Copy)]
struct Handle<'a>(&'a (Vec<Cell>, Vec<Cell>, Vec<Payload>));
impl<'a> Read<'a> for Handle<'a> {{
    type Error = String;
    fn value_type(self, value: Cell) -> Type {{ value + 1000 }}
    fn opcode(self) -> Option<Code> {{ Some(Code::Pair) }}
    fn results(self) -> &'a [Cell] {{ &self.0.0 }}
    fn inputs(self) -> &'a [Cell] {{ &self.0.1 }}
    fn fields(self) -> &'a [Payload] {{ &self.0.2 }}
    fn error(self, message: &str) -> String {{ message.to_owned() }}
}}
fn main() {{
    text::roundtrip();
    signatures::check();
    let mut store = Vec::new();
    let id = Sink {{ store: &mut store }}.pair(10, 20, 30, 40, Tag {{ bits: 99 }});
    assert_eq!(store[id], (vec![10, 20], vec![40, 30], vec![Payload::Tag(Tag {{ bits: 99 }})]));
    Handle(&store[id]).validate(&Limits(100)).unwrap();
    let View::Pair(pair) = Handle(&store[id]).view();
    assert_eq!((pair.low, pair.high, pair.left, pair.right, pair.tag.map(|tag| tag.bits)), (10, 20, 30, 40, Some(99)));
    assert_eq!(Handle(&store[id]).summary().unwrap().bits, 99);
    assert_eq!(Handle(&store[id]).summary().unwrap().ty, 1030);
    let mut visited = Vec::new();
    Handle(&store[id]).try_visit_ownership::<core::convert::Infallible>(|reg, moved| {{
        visited.push((reg, moved)); Ok(())
    }}).unwrap();
    assert_eq!(visited, vec![(30, true), (40, false)]);
    store[id].2[0] = Payload::Tag(Tag {{ bits: 101 }});
    assert_eq!(Handle(&store[id]).validate(&Limits(100)).unwrap_err(), "tag exceeds limit");
    store[id].2[0] = Payload::Tag(Tag {{ bits: 0 }});
    assert_eq!(Handle(&store[id]).validate(&Limits(100)).unwrap_err(), "zero tag");
    store[id].2.clear();
    assert!(Handle(&store[id]).validate(&Limits(100)).is_err());
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
            && all(token(7).number(), token(7).number(), |a, b| a == b)
            && (token(7).number()? > 0 || token(0).number()? > 0),
        next: token(7).number(),
    },
    mnemonic: "check", storage: Empty {},
}
"#;
    let generated = common::raw_plan(defs).unwrap().generate();
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
    fn lane_count(&self) -> u32 {{ 1 }}
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
    let error = common::raw_parse(
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
