use std::collections::{BTreeMap, BTreeSet};

use veloc_semantics::{BvConst, BvOp};

use crate::builtins::Builtins;
use crate::syntax::{Kind, Node, Record};
use crate::type_set::TypeSet;
use crate::types::Types;
use crate::{Error, storage};

#[path = "operation.rs"]
mod operation;

/// Checked operation definitions, independent of the runtime MIR.
pub struct Definitions {
    pub(crate) encoding: crate::encoding::TypeEncoding,
    pub(crate) builtins: Builtins,
    pub(crate) types: Types,
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
    Class(TypeSet),
    Exact(String),
    Bind(u8, TypeSet),
    Same(u8),
    ElementOf(u8),
    VectorOf(u8),
    ShapeOf(u8, TypeSet),
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
    pub packing: BTreeMap<String, Binding>,
    pub signature_source: Option<SignatureSource>,
    pub text: Option<Node>,
    pub traits: Vec<String>,
    pub memory: String,
    pub constraints: Vec<String>,
    pub identity: Option<BvConst>,
    pub absorbing: Option<BvConst>,
    pub semantics: Option<Semantic>,
}

#[derive(Debug)]
pub(crate) enum Binding {
    Name(String),
    Array(Vec<Binding>),
    Pool(String),
    Table { cases: String, default: String },
}

pub(crate) enum SignatureSource {
    Function(String),
    Signature(String),
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

struct Variable {
    slot: u8,
    class: TypeSet,
    possible: TypeSet,
    bound: bool,
}

pub(crate) fn parse(source: &str) -> Result<Definitions, Error> {
    let records = crate::syntax::parse(source)?;
    let mut names = BTreeSet::new();
    for record in &records {
        if !names.insert((record.kind.clone(), record.name.clone())) {
            return Err(Error::at(
                source,
                record.offset,
                format!("duplicate {} `{}`", record.kind, record.name),
            ));
        }
        identifier(source, record.offset, &record.name)?;
    }
    let encoding = crate::encoding::TypeEncoding::compile(&records, source)?;
    let types = Types::compile(&records, source, &encoding)?;
    let builtins = Builtins::compile(&records, source)?;
    let storage = storage::compile(&records, source)?;
    let mut ops = Vec::new();
    for record in records {
        match record.kind.as_str() {
            "op" => ops.push(operation::parse(
                source, record, &storage, &types, &builtins,
            )?),
            "format" | "layout" | "record" | "encoding" => {}
            kind if Builtins::is_definition(kind) => {}
            kind if Types::is_definition(kind) => {}
            _ => {
                return Err(Error::at(
                    source,
                    record.offset,
                    format!("unknown definition kind `{}`", record.kind),
                ));
            }
        }
    }
    let definitions = Definitions {
        encoding,
        builtins,
        types,
        storage,
        ops,
    };
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
            operation::validate_packing(source, op, format, &self.storage)?;
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
            crate::semantic::validate(source, op, &self.types, &self.builtins)?;
        }
        Ok(())
    }
}

fn pattern(
    source: &str,
    node: Node,
    variables: &mut BTreeMap<String, Variable>,
    types: &Types,
) -> Result<Pattern, Error> {
    match node.kind {
        Kind::Name(name) if types.exact.contains_key(&name) => Ok(Pattern::Exact(name)),
        Kind::Name(ref name) if types.classes.contains_key(name) => {
            Ok(Pattern::Class(types.set(source, &node)?))
        }
        Kind::Union(_) | Kind::Intersection(_) => Ok(Pattern::Class(types.set(source, &node)?)),
        Kind::Call(ref kind, _) if kind == "vectors" => {
            Ok(Pattern::Class(types.set(source, &node)?))
        }
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
                Some(types.set(source, &args.next().unwrap())?)
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
            match kind.as_str() {
                "element" => binding.possible.retain_shapes(!1),
                "vector" => binding.possible.intersect(&types.lanes),
                "shape" => binding
                    .possible
                    .retain_shapes(class.as_ref().unwrap().shapes()),
                _ => unreachable!("type pattern kind has been checked"),
            }
            if binding.possible.is_empty() {
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

pub(crate) struct Fields<'a> {
    source: &'a str,
    offset: usize,
    name: String,
    fields: BTreeMap<String, Node>,
}

impl<'a> Fields<'a> {
    pub(crate) fn new(source: &'a str, record: Record) -> Self {
        Self {
            source,
            offset: record.offset,
            name: record.name,
            fields: record.fields,
        }
    }
    pub(crate) fn error(&self, message: impl Into<String>) -> Error {
        Error::at(self.source, self.offset, message)
    }
    pub(crate) fn take(&mut self, name: &str) -> Result<Node, Error> {
        self.fields
            .remove(name)
            .ok_or_else(|| self.error(format!("{} is missing `{name}`", self.name)))
    }
    pub(crate) fn optional(&mut self, name: &str) -> Option<Node> {
        self.fields.remove(name)
    }
    pub(crate) fn finish(&self) -> Result<(), Error> {
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

pub(crate) fn name(source: &str, node: Node) -> Result<String, Error> {
    match node.kind {
        Kind::Name(name) => Ok(name),
        _ => Err(Error::at(source, node.offset, "expected a name")),
    }
}

pub(crate) fn list(source: &str, node: Node) -> Result<Vec<Node>, Error> {
    match node.kind {
        Kind::List(values) => Ok(values),
        _ => Err(Error::at(source, node.offset, "expected a list")),
    }
}

fn algebraic_constant(source: &str, node: Node) -> Result<BvConst, Error> {
    let offset = node.offset;
    let name = name(source, node)?;
    BvConst::from_name(&name).ok_or_else(|| {
        Error::at(
            source,
            offset,
            format!("unknown algebraic constant `{name}`"),
        )
    })
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
    use crate::fixtures::parse;

    const SOURCE: &str = r#"
        format Binary {
            fields: [opcode(Opcode), args(values(2))],
            opcode: dynamic(opcode)
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
