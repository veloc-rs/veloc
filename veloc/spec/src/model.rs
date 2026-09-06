use std::collections::{BTreeMap, BTreeSet};

use veloc_semantics::{BvConst, BvOp};

use crate::syntax::{Kind, Node, Record};
use crate::{Error, storage};

#[path = "operation.rs"]
mod operation;

/// Checked operation definitions, independent of the runtime MIR.
pub struct Definitions {
    pub(crate) storage: storage::Storage,
    pub(crate) ops: Vec<Op>,
}

impl Definitions {
    pub fn operation_count(&self) -> usize {
        self.ops.len()
    }
    pub fn format_count(&self) -> usize {
        self.storage.formats.len()
    }
}

pub(crate) struct TypeDef {
    pub operands: TypeList,
    pub results: TypeList,
    pub relations: Vec<Relation>,
}

pub(crate) enum TypeList {
    Fixed(Vec<Pattern>),
    Variadic(Vec<Pattern>),
    Signature,
}

impl TypeList {
    pub fn patterns(&self) -> Option<&[Pattern]> {
        match self {
            Self::Fixed(patterns) => Some(patterns),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Pattern {
    Class(String),
    Exact(String),
    Bind(u8, String),
    Same(u8),
    ElementOf(u8),
    VectorOf(u8),
    ShapeOf(u8, String),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Slot {
    pub result: bool,
    pub index: u8,
}

pub(crate) struct Relation {
    pub kind: String,
    pub lhs: Slot,
    pub rhs: Slot,
}

pub(crate) struct Op {
    pub offset: usize,
    pub name: String,
    pub mnemonic: String,
    pub format: String,
    pub signature: TypeDef,
    pub params: Vec<Param>,
    pub packing: BTreeMap<String, Vec<String>>,
    pub traits: Vec<String>,
    pub memory: String,
    pub constraints: Vec<String>,
    pub identity: Option<String>,
    pub absorbing: Option<String>,
    pub semantics: Option<Semantic>,
}

pub(crate) struct Param {
    pub name: String,
    pub kind: ParamKind,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ParamKind {
    Value,
    Property(String),
    Values,
    Successor,
    Successors,
}

pub(crate) struct Semantic {
    pub steps: Vec<SemanticStep>,
    pub output: u16,
    pub inputs: u8,
}

pub(crate) enum SemanticStep {
    Input(u8),
    Const(BvConst),
    Apply { op: BvOp, args: Vec<u16> },
}

impl Semantic {
    pub(crate) fn primitive(&self) -> Option<BvOp> {
        let SemanticStep::Apply { op, args } = &self.steps[self.output as usize] else {
            return None;
        };
        (args.len() == self.inputs as usize
            && args.iter().enumerate().all(|(i, &arg)| {
                matches!(self.steps[arg as usize], SemanticStep::Input(index) if index as usize == i)
            }))
        .then_some(*op)
    }
}

impl Op {
    pub fn method_name(&self) -> String {
        self.mnemonic.replace('-', "_")
    }
}

const CLASSES: &[&str] = &[
    "Any",
    "Scalar",
    "ScalarInteger",
    "ScalarIntegerOrPointer",
    "ScalarFloat",
    "Integer",
    "IntegerOrBool",
    "Float",
    "Number",
    "Vector",
    "IntegerVector",
];
const EXACT: &[&str] = &["I8", "I16", "I32", "I64", "F32", "F64", "BOOL", "PTR"];
const TRAITS: &[&str] = &[
    "TERMINATOR",
    "COMMUTATIVE",
    "MAY_TRAP",
    "ASSOCIATIVE",
    "IDEMPOTENT",
];
const MEMORY: &[&str] = &[
    "NONE",
    "UNKNOWN",
    "HEAP_READ",
    "HEAP_WRITE",
    "STACK_READ",
    "STACK_WRITE",
    "GLOBAL_READ",
    "GLOBAL_WRITE",
    "TABLE_READ",
    "TABLE_WRITE",
];
const CONSTANTS: &[&str] = &["Zero", "One", "AllOnes"];

struct Variable {
    slot: u8,
    class: String,
    domain: u8,
    bound: bool,
}

// Scalar integer/float/bool/pointer and vector integer/float/bool domains.
// This checks impossible schemes, not concrete widths or target legality.
fn class_domain(class: &str) -> u8 {
    match class {
        "Any" => 0b111_1111,
        "Scalar" => 0b000_0111,
        "ScalarInteger" => 0b000_0001,
        "ScalarIntegerOrPointer" => 0b000_1001,
        "ScalarFloat" => 0b000_0010,
        "Integer" => 0b001_0001,
        "IntegerOrBool" => 0b101_0101,
        "Float" => 0b010_0010,
        "Number" => 0b011_0011,
        "Vector" => 0b111_0000,
        "IntegerVector" => 0b001_0000,
        _ => unreachable!("type classes have been checked"),
    }
}

pub(crate) fn parse(source: &str) -> Result<Definitions, Error> {
    let records = crate::syntax::parse(source)?;
    let storage = storage::compile(&records, source)?;
    let mut ops = Vec::new();
    let mut names = BTreeSet::new();
    for record in records {
        if !names.insert((record.kind.clone(), record.name.clone())) {
            return Err(Error::at(
                source,
                record.offset,
                format!("duplicate {} `{}`", record.kind, record.name),
            ));
        }
        identifier(source, record.offset, &record.name)?;
        match record.kind.as_str() {
            "op" => ops.push(operation::parse(source, record)?),
            "format" | "layout" => {}
            _ => {
                return Err(Error::at(
                    source,
                    record.offset,
                    format!("unknown definition kind `{}`", record.kind),
                ));
            }
        }
    }
    let definitions = Definitions { storage, ops };
    definitions.validate(source)?;
    Ok(definitions)
}

impl Definitions {
    fn validate(&self, source: &str) -> Result<(), Error> {
        let mut mnemonics = BTreeSet::new();
        let mut methods = BTreeMap::new();
        for op in &self.ops {
            let fail = |message| Error::at(source, op.offset, message);
            if !mnemonics.insert(&op.mnemonic) {
                return Err(fail(format!("duplicate mnemonic `{}`", op.mnemonic)));
            }
            if op.mnemonic.is_empty()
                || !op
                    .mnemonic
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
            {
                return Err(fail(format!("invalid mnemonic `{}`", op.mnemonic)));
            }
            let method = op.method_name();
            if let Some(previous) = methods.insert(method.clone(), &op.mnemonic) {
                return Err(fail(format!(
                    "mnemonics `{previous}` and `{}` produce the same method name `{method}`",
                    op.mnemonic
                )));
            }
            let format = self
                .storage
                .formats
                .iter()
                .find(|f| f.name == op.format)
                .ok_or_else(|| fail(format!("unknown format `{}`", op.format)))?;
            if let Some(fixed) = &format.fixed_opcode
                && fixed != &op.name
            {
                return Err(fail(format!(
                    "format `{}` has fixed opcode `{fixed}`, not `{}`",
                    op.format, op.name
                )));
            }
            let ty = &op.signature;
            operation::validate_packing(source, op, format)?;
            match (format.arity, &ty.operands) {
                (Some(arity), TypeList::Fixed(patterns)) if arity == patterns.len() => {}
                (None, TypeList::Variadic(_)) => {}
                _ => {
                    return Err(fail(
                        "storage operands do not match the logical signature".into(),
                    ));
                }
            }
            for constraint in &op.constraints {
                let expected = match constraint.as_str() {
                    "PointerComparison" => "IntCompare",
                    "NonZeroScale" => "PtrIndex",
                    "VectorConstant" => "Vconst",
                    "ShuffleMask" => "Shuffle",
                    _ => return Err(fail(format!("unknown constraint `{constraint}`"))),
                };
                if op.format != expected {
                    return Err(fail(format!(
                        "constraint `{constraint}` requires format `{expected}`"
                    )));
                }
            }
            if (op.identity.is_some()
                || op.absorbing.is_some()
                || op.traits.iter().any(|t| t == "IDEMPOTENT"))
                && !["ASSOCIATIVE", "COMMUTATIVE"]
                    .iter()
                    .all(|required| op.traits.iter().any(|t| t == required))
            {
                return Err(fail(
                    "algebraic shortcuts require associative and commutative operations".into(),
                ));
            }
            crate::semantic::validate(source, op)?;
        }
        Ok(())
    }
}

fn pattern(
    source: &str,
    node: Node,
    variables: &mut BTreeMap<String, Variable>,
) -> Result<Pattern, Error> {
    match node.kind {
        Kind::Name(name) if EXACT.contains(&name.as_str()) => Ok(Pattern::Exact(name)),
        Kind::Name(name) if CLASSES.contains(&name.as_str()) => Ok(Pattern::Class(name)),
        Kind::Name(name) => {
            let var = variables.get_mut(&name).ok_or_else(|| {
                Error::at(
                    source,
                    node.offset,
                    format!("unbound type variable `{name}`"),
                )
            })?;
            if var.bound {
                Ok(Pattern::Same(var.slot))
            } else {
                var.bound = true;
                Ok(Pattern::Bind(var.slot, var.class.clone()))
            }
        }
        Kind::Call(kind, args) => {
            let expected = match kind.as_str() {
                "shape" => 2,
                "element" | "vector" => 1,
                _ => {
                    return Err(Error::at(
                        source,
                        node.offset,
                        format!("unknown type pattern `{kind}`"),
                    ));
                }
            };
            if args.len() != expected {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("{kind} expects {expected} arguments"),
                ));
            }
            let mut args = args.into_iter();
            let variable = name(source, args.next().unwrap())?;
            let class = if expected == 2 {
                Some(choice(source, args.next().unwrap(), CLASSES, "type class")?)
            } else {
                None
            };
            let binding = variables
                .get_mut(&variable)
                .filter(|var| var.bound)
                .ok_or_else(|| {
                    Error::at(
                        source,
                        node.offset,
                        format!("unbound type variable `{variable}`"),
                    )
                })?;
            let domain = match kind.as_str() {
                "element" => class_domain("Vector"),
                "vector" => class_domain("Scalar"),
                "shape" => {
                    let class = class_domain(class.as_deref().unwrap());
                    let scalars = if class & 0b000_1111 != 0 {
                        0b000_1111
                    } else {
                        0
                    };
                    let vectors = if class & 0b111_0000 != 0 {
                        0b111_0000
                    } else {
                        0
                    };
                    scalars | vectors
                }
                _ => class_domain("Any"),
            };
            binding.domain &= domain;
            if binding.domain == 0 {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("impossible {kind} constraint on type variable `{variable}`"),
                ));
            }
            let var = binding.slot;
            Ok(match kind.as_str() {
                "element" => Pattern::ElementOf(var),
                "vector" => Pattern::VectorOf(var),
                "shape" => Pattern::ShapeOf(var, class.unwrap()),
                _ => unreachable!(),
            })
        }
        _ => Err(Error::at(source, node.offset, "unknown type pattern")),
    }
}

struct Fields<'a> {
    source: &'a str,
    offset: usize,
    name: String,
    fields: BTreeMap<String, Node>,
}

impl<'a> Fields<'a> {
    fn new(source: &'a str, record: Record) -> Self {
        Self {
            source,
            offset: record.offset,
            name: record.name,
            fields: record.fields,
        }
    }
    fn error(&self, message: impl Into<String>) -> Error {
        Error::at(self.source, self.offset, message)
    }
    fn take(&mut self, name: &str) -> Result<Node, Error> {
        self.fields
            .remove(name)
            .ok_or_else(|| self.error(format!("{} is missing `{name}`", self.name)))
    }
    fn optional(&mut self, name: &str) -> Option<Node> {
        self.fields.remove(name)
    }
    fn finish(&self) -> Result<(), Error> {
        if let Some((field, node)) = self.fields.first_key_value() {
            Err(Error::at(
                self.source,
                node.offset,
                format!("unknown field `{field}`"),
            ))
        } else {
            Ok(())
        }
    }
}

fn name(source: &str, node: Node) -> Result<String, Error> {
    match node.kind {
        Kind::Name(name) => Ok(name),
        _ => Err(Error::at(source, node.offset, "expected a name")),
    }
}

fn list(source: &str, node: Node) -> Result<Vec<Node>, Error> {
    match node.kind {
        Kind::List(values) => Ok(values),
        _ => Err(Error::at(source, node.offset, "expected a list")),
    }
}

fn choice(source: &str, node: Node, choices: &[&str], what: &str) -> Result<String, Error> {
    let offset = node.offset;
    let name = name(source, node)?;
    if choices.contains(&name.as_str()) {
        Ok(name)
    } else {
        Err(Error::at(
            source,
            offset,
            format!("unknown {what} `{name}`"),
        ))
    }
}

fn choices(source: &str, node: Node, choices: &[&str], what: &str) -> Result<Vec<String>, Error> {
    let offset = node.offset;
    let values = list(source, node)?
        .into_iter()
        .map(|n| choice(source, n, choices, what))
        .collect::<Result<Vec<_>, _>>()?;
    if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
        return Err(Error::at(source, offset, format!("duplicate {what}")));
    }
    Ok(values)
}

pub(crate) fn identifier(source: &str, offset: usize, name: &str) -> Result<(), Error> {
    const KEYWORDS: &[&str] = &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
        "true", "type", "unsafe", "use", "where", "while", "abstract", "become", "box", "do",
        "final", "gen", "macro", "override", "priv", "try", "typeof", "unsized", "virtual",
        "yield",
    ];
    if name == "_"
        || name.is_empty()
        || KEYWORDS.contains(&name)
        || !name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        Err(Error::at(
            source,
            offset,
            format!("invalid generated identifier `{name}`"),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    const SOURCE: &str = r#"
        format Binary {
            fields: [opcode(Opcode), args(values(2))],
            opcode: dynamic(opcode), text: values(2)
        }
        op Add<T: Integer>(lhs: T, rhs: T) -> (result: T) {
            mnemonic: "i-add", storage: Binary { args: [lhs, rhs] },
            traits: [], memory: NONE
        }
    "#;

    #[test]
    fn method_names_come_from_mnemonics() {
        let defs = parse(SOURCE).unwrap();
        assert_eq!(defs.ops[0].method_name(), "i_add");
        assert_eq!(defs.ops[0].mnemonic, "i-add");
    }

    #[test]
    fn rejects_normalized_method_name_collisions() {
        let source = format!(
            "{SOURCE}\n\
             op Other<T: Integer>(lhs: T, rhs: T) -> (result: T) {{ mnemonic: \"i_add\", storage: Binary {{ args: [lhs, rhs] }}, traits: [], memory: NONE }}"
        );
        let error = match parse(&source) {
            Ok(_) => panic!("colliding generated method names were accepted"),
            Err(error) => error,
        };
        assert!(error.message.contains("same method name `i_add`"));
    }

    #[test]
    fn removed_builder_fields_are_unknown_fields() {
        for builder in ["iadd", "iadd(args)"] {
            let source =
                SOURCE.replace("memory: NONE", &format!("memory: NONE, builder: {builder}"));
            let error = match parse(&source) {
                Ok(_) => panic!("removed builder field was accepted"),
                Err(error) => error,
            };
            assert!(error.message.contains("unknown field `builder`"));
        }
    }

    #[test]
    fn method_identifier_checks_are_left_to_the_emitter() {
        let defs = parse(&SOURCE.replace("i-add", "return")).unwrap();
        assert_eq!(defs.ops[0].method_name(), "return");
    }
}
