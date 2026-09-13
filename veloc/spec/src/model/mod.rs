//! Checked operation contracts, independent of runtime IR containers.

pub(crate) mod comparisons;
pub(crate) mod constraints;
pub(crate) mod data;
pub(crate) mod encoding;
pub(crate) mod expr;
pub(crate) mod metadata;
pub(crate) mod records;
use std::collections::{BTreeMap, BTreeSet};

use veloc_semantics::{BvConst, BvOp};

use crate::model::encoding::Encodings;
use crate::syntax::{Kind, Node, Record};
use crate::types::TypeSet;
use crate::types::Types;
use crate::{Error, storage};

mod operation;

/// Checked operation definitions, independent of the runtime MIR.
pub struct Definitions {
    pub(crate) encodings: Encodings,
    pub(crate) data: crate::model::data::Types,
    pub(crate) comparisons: Vec<crate::model::comparisons::Comparison>,
    pub(crate) types: Types,
    pub(crate) storage: storage::Storage,
    pub(crate) ops: Vec<Op>,
    pub(crate) properties: Vec<Property>,
    pub(crate) expressions: expr::Library,
}

/// Shared names available to operation contracts and pure projections.
#[derive(Clone, Copy)]
pub(crate) struct Vocabulary<'a> {
    pub types: &'a Types,
    pub encodings: &'a Encodings,
    pub data: &'a data::Types,
    pub comparisons: &'a [comparisons::Comparison],
}

pub(crate) struct Property {
    pub offset: usize,
    pub name: String,
    pub constraints: Vec<constraints::Constraint>,
}

impl Definitions {
    pub fn operation_count(&self) -> usize {
        self.ops.len()
    }
    pub fn format_count(&self) -> usize {
        match &self.storage.strategy {
            storage::Strategy::Packed => self.storage.formats.len(),
            storage::Strategy::Operands(operands) => operands.format_count(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TypeDef {
    pub operands: TypeList,
    pub results: TypeList,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Pattern {
    /// A typed property's type; the set permits type-only validation as well.
    Property(String, TypeSet),
    Callable,
    Set(TypeSet),
    Exact(String),
    Bind(u8, TypeSet),
    Same(u8),
    ElementOf(u8),
    VectorOf(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Slot {
    pub result: bool,
    pub index: u8,
}

pub(crate) struct Op {
    pub offset: usize,
    pub name: String,
    pub mnemonic: String,
    pub meta: crate::model::metadata::Metadata,
    pub format: String,
    pub signature: TypeDef,
    pub params: Vec<Param>,
    pub projection: Projection,
    pub signature_source: Option<SignatureSource>,
    pub text: Option<Node>,
    pub traits: BTreeSet<String>,
    pub queries: BTreeMap<String, expr::Expr>,
    pub constraints: Vec<crate::model::constraints::Constraint>,
    pub identity: Option<BvConst>,
    pub absorbing: Option<BvConst>,
    pub semantics: Option<Semantic>,
}

pub(crate) enum Projection {
    Packed(BTreeMap<String, Binding>),
    Operands(crate::storage::operands::Projection),
}

impl Op {
    pub(crate) fn bindings(&self) -> &BTreeMap<String, Binding> {
        match &self.projection {
            Projection::Packed(bindings) => bindings,
            Projection::Operands(_) => panic!("packed emitter requires field bindings"),
        }
    }

    pub(crate) fn operands(&self) -> &crate::storage::operands::Projection {
        match &self.projection {
            Projection::Operands(projection) => projection,
            Projection::Packed(_) => panic!("operand emitter requires an operand projection"),
        }
    }
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
    Value(String),
}

pub(crate) struct Param {
    pub moves: bool,
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
    pub outputs: Vec<u16>,
    pub inputs: u8,
    pub properties: Vec<String>,
    pub traps: Vec<(u16, veloc_semantics::Trap)>,
    pub instances: Vec<crate::semantic::Instance>,
}

pub(crate) type SemanticStep = veloc_semantics::Step<Vec<u16>>;

impl Semantic {
    pub fn program(&self) -> veloc_semantics::Program<'_, Vec<u16>> {
        veloc_semantics::Program {
            inputs: self.inputs,
            properties: self.properties.len() as u8,
            steps: &self.steps,
            outputs: &self.outputs,
            traps: &self.traps,
        }
    }
    pub(crate) fn primitive(&self) -> Option<BvOp> {
        self.program().primitive()
    }
}

impl Op {
    pub fn method_name(&self) -> String {
        crate::storage::constructor_name(&self.mnemonic.replace('-', "_"))
    }
}

struct Variable {
    slot: u8,
    set: TypeSet,
    possible: TypeSet,
    bound: bool,
}

pub(crate) fn parse(source: &str) -> Result<Definitions, Error> {
    let records = crate::syntax::parse(source)?;
    from_records(source, records)
}

pub(crate) fn from_records(source: &str, records: Vec<Record>) -> Result<Definitions, Error> {
    let mut names = BTreeSet::new();
    for record in &records {
        if !names.insert((record.kind.clone(), record.name.clone())) {
            return Err(Error::at(
                source,
                record.offset,
                format!("duplicate {} `{}`", record.kind, record.name),
            ));
        }
        if matches!(record.kind.as_str(), "fn" | "const") {
            for part in record.name.split("::") {
                identifier(source, record.offset, part)?;
            }
        } else {
            identifier(source, record.offset, &record.name)?;
        }
    }
    let types = Types::compile(&records, source)?;
    let encodings = encoding::compile(&records, source)?;
    let comparisons = crate::model::comparisons::compile(&records, source)?;
    let data = crate::model::data::Types::compile(&records, source)?;
    let storage = storage::compile(&records, source, &data)?;
    let vocabulary = Vocabulary {
        types: &types,
        encodings: &encodings,
        data: &data,
        comparisons: &comparisons,
    };
    let mut expressions = expr::Library::compile(&records, source, vocabulary)?;
    let mut ops = Vec::new();
    let mut properties = Vec::new();
    for record in records {
        match record.kind.as_str() {
            "property" => {
                if !matches!(record.name.as_str(), "VectorConst" | "Int" | "Float") {
                    return Err(Error::at(source, record.offset, "unknown typed property"));
                }
                let offset = record.offset;
                let name = record.name.clone();
                let mut fields = Fields::new(source, record);
                let nodes = Some(fields.take("verify")?);
                let constraints = constraints::check_property(
                    source,
                    &name,
                    nodes,
                    vocabulary,
                    &mut expressions,
                )?;
                fields.finish()?;
                properties.push(Property {
                    offset,
                    name,
                    constraints,
                });
            }
            "op" => ops.push(operation::parse(
                source,
                record,
                &storage,
                vocabulary,
                &mut expressions,
            )?),
            "layout" | "struct" | "enum" | "encoding" | "comparison" | "storage" | "fn"
            | "const" => {}
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
    let mut definitions = Definitions {
        encodings,
        data,
        comparisons,
        types,
        storage,
        ops,
        properties,
        expressions,
    };
    definitions.validate(source)?;
    Ok(definitions)
}

impl Definitions {
    fn validate(&mut self, source: &str) -> Result<(), Error> {
        let mut mnemonics = BTreeSet::new();
        let mut methods = BTreeMap::new();
        let meta_type = crate::model::metadata::record_type(&self.ops).map(str::to_owned);
        for op in &mut self.ops {
            if Some(op.meta.name.as_str()) != meta_type.as_deref() {
                return Err(Error::at(
                    source,
                    op.offset,
                    "all operations in a unit must use the same metadata struct type",
                ));
            }
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
            if let storage::Strategy::Packed = self.storage.strategy {
                let format = self
                    .storage
                    .formats
                    .iter()
                    .find(|f| f.name == op.format)
                    .ok_or_else(|| fail(format!("unknown format `{}`", op.format)))?;
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
            }
            if (op.identity.is_some() || op.absorbing.is_some() || op.traits.contains("IDEMPOTENT"))
                && !["ASSOCIATIVE", "COMMUTATIVE"]
                    .iter()
                    .all(|name| op.traits.contains(*name))
            {
                return Err(fail(
                    "algebraic shortcuts require associative and commutative operations".into(),
                ));
            }
            let instances = crate::semantic::validate(source, op, &self.types)?;
            if let Some(sem) = &mut op.semantics {
                sem.instances = instances;
            }
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
    if let Some(name) = types.exact_name(&node) {
        return Ok(Pattern::Exact(name));
    }
    match node.kind {
        Kind::Name(ref name) if name.contains("::") => Err(Error::at(
            source,
            node.offset,
            "unknown type constant or undeclared Type",
        )),
        Kind::Name(name) if name == "Callable" => Ok(Pattern::Callable),
        Kind::Name(ref name) if types.sets.contains_key(name) => {
            Ok(Pattern::Set(types.set(source, &node)?))
        }
        Kind::Union(_) | Kind::Intersection(_) => Ok(Pattern::Set(types.set(source, &node)?)),
        Kind::Call(ref kind, _) if kind == "vectors" => Ok(Pattern::Set(types.set(source, &node)?)),
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
                Ok(Pattern::Bind(var.slot, var.set.clone()))
            }
        }
        Kind::Call(kind, args) => {
            let expected = match kind.as_str() {
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
        struct Binary {
            args: values(2),
        }
        op Add<T: Integer>(lhs: T, rhs: T) -> (result: T) {
    meta: OpInfo { traits: OpTraits::empty(), memory: MemoryEffect::NONE },
            mnemonic: "i-add", storage: Binary { args: [lhs, rhs] },
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
             op Other<T: Integer>(lhs: T, rhs: T) -> (result: T) {{ meta: OpInfo {{ traits: OpTraits::empty(), memory: MemoryEffect::NONE }}, mnemonic: \"i_add\", storage: Binary {{ args: [lhs, rhs] }},  }}"
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
            let source = SOURCE.replace("mnemonic:", &format!("builder: {builder}, mnemonic:"));
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
        assert_eq!(defs.ops[0].method_name(), "ret");
    }
}
