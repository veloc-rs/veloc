//! Structured logical properties and their generated Rust representation.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::syntax::{Decl, DeclKind, Kind, Node};
use crate::{Error, model};

/// Logical reference structure, independent of payload placement.
/// Lists compose with leaves and edges instead of defining separate role labels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum References {
    #[default]
    Data,
    Operand,
    List(Box<Self>),
    Edge(Box<Self>),
}

impl References {
    pub fn is_data(&self) -> bool {
        match self {
            Self::Data => true,
            Self::List(item) => item.is_data(),
            Self::Operand | Self::Edge(_) => false,
        }
    }
    pub fn is_operand(&self) -> bool {
        matches!(self, Self::Operand)
    }
    pub fn is_operands(&self) -> bool {
        matches!(self, Self::List(item) if item.is_operand())
    }
    pub fn is_edge(&self) -> bool {
        matches!(self, Self::Edge(_))
    }
    pub fn is_edges(&self) -> bool {
        matches!(self, Self::List(item) if item.is_edge())
    }
    pub fn arity(&self) -> Option<usize> {
        if self.is_data() {
            Some(0)
        } else if self.is_operand() {
            Some(1)
        } else {
            None
        }
    }
    fn parse(
        source: &str,
        node: &Node,
        records: &[Decl],
        active: &mut BTreeSet<String>,
    ) -> Result<Self, Error> {
        match &node.kind {
            Kind::Name(name) if name == "operand" => Ok(Self::Operand),
            Kind::Name(name) => {
                let record = records
                    .iter()
                    .find(|r| r.name == *name && rust_binding(r).is_some())
                    .ok_or_else(|| {
                        Error::at(
                            source,
                            node.offset,
                            "field reference requires a declared Rust type",
                        )
                    })?;
                if !active.insert(name.clone()) {
                    return Err(Error::at(source, node.offset, "cyclic field reference"));
                }
                let result = match record.fields.get("field") {
                    Some(field) => Self::parse(source, field, records, active),
                    None => Ok(Self::Data),
                };
                active.remove(name);
                result
            }
            Kind::Call(name, args) if matches!(name.as_str(), "list" | "edge") => {
                let [item] = args.as_slice() else {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "field constructor requires one element type",
                    ));
                };
                let item = Self::parse(source, item, records, active)?;
                if name == "edge" {
                    if !item.is_operand() {
                        return Err(Error::at(
                            source,
                            node.offset,
                            "edge parameters require an SSA operand element",
                        ));
                    }
                    Ok(Self::Edge(Box::new(item)))
                } else {
                    Ok(Self::List(Box::new(item)))
                }
            }
            _ => Err(Error::at(
                source,
                node.offset,
                "expected operand, a field type, list(type), or edge(type)",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Placement {
    #[default]
    Auto,
    Pooled,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Policy {
    pub borrowed: bool,
    pub references: References,
    pub storage: Placement,
}

impl Policy {
    pub fn parse(source: &str, record: &Decl, records: &[Decl]) -> Result<Self, Error> {
        let mut fields = model::Fields::new(source, record.clone());
        fields.optional("trait");
        fields.optional("analysis");
        let borrowed = match fields.optional("view") {
            None => false,
            Some(node) => match model::name(source, node)?.as_str() {
                "borrowed" => true,
                "copied" => false,
                _ => return Err(fields.error("expected borrowed or copied view")),
            },
        };
        let references = match fields.optional("field") {
            None => References::Data,
            Some(node) => References::parse(source, &node, records, &mut BTreeSet::new())?,
        };
        // Rust-bound views must implement one of the supported access contracts.
        // Structural records compose separately; do not silently lose nested uses.
        if !(references.is_data()
            || references.is_operand()
            || references.is_operands()
            || references.is_edge()
            || references.is_edges())
        {
            return Err(fields.error("unsupported nested Rust field interface"));
        }
        let storage = match fields.optional("storage") {
            None => Placement::Auto,
            Some(node) => match model::name(source, node)?.as_str() {
                "auto" => Placement::Auto,
                "pooled" => Placement::Pooled,
                _ => return Err(fields.error("expected auto or pooled storage")),
            },
        };
        fields.finish()?;
        Ok(Self {
            borrowed,
            references,
            storage,
        })
    }
}

/// Rust type bindings are nominal in defs; paths only control code emission.
#[derive(Debug, Clone, Default)]
pub(crate) struct RustTypes {
    external: crate::interfaces::Bindings,
    pub analysis: Option<super::metadata::Analysis>,
    policies: BTreeMap<String, Policy>,
}

pub(crate) use crate::interfaces::{primitive, rust_binding, rust_path};

impl RustTypes {
    pub fn compile(records: &[Decl], source: &str) -> Result<Self, Error> {
        let external = crate::interfaces::Bindings::compile(records, source)?;
        let mut policies = BTreeMap::new();
        for record in records.iter().filter(|r| rust_binding(r).is_some()) {
            policies.insert(record.name.clone(), Policy::parse(source, record, records)?);
        }
        Ok(Self {
            external,
            policies,
            analysis: super::metadata::Analysis::compile(records, source)?,
        })
    }
    pub fn method_trait(&self, name: &str, is_const: bool) -> String {
        self.external
            .method_trait(name, "crate::type_methods", is_const)
    }
    pub fn policy(&self, name: &str) -> Policy {
        self.policies.get(name).cloned().unwrap_or_default()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.external.0.contains_key(name)
    }

    pub fn rust(&self, name: &str) -> String {
        self.external
            .0
            .get(name)
            .map(|b| b.path.clone())
            .unwrap_or_else(|| name.to_owned())
    }

    pub fn qualified(&self, name: &str) -> String {
        if self.contains(name) || primitive(name) {
            self.rust(name)
        } else {
            format!("crate::inst::{name}")
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RecordDef {
    pub name: String,
    pub fields: Vec<RecordField>,
}

#[derive(Debug, Clone)]
pub(crate) struct RecordField {
    pub name: String,
    pub ty: PropertyType,
    pub rust: String,
    // A logical SSA refinement is checked in expressions but erased in storage.
    pub logical: Option<Node>,
    pub policy: Policy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PropertyType {
    Named(String),
    Optional(String),
    Sequence(String),
    Values(usize),
    Array(String, usize),
}

impl PropertyType {
    pub fn rust(&self, types: &RustTypes) -> String {
        match self {
            Self::Named(ty) => types.rust(ty),
            Self::Optional(ty) => format!("Option<{}>", types.rust(ty)),
            Self::Sequence(ty) => format!("&'a [{}]", types.rust(ty)),
            Self::Array(ty, n) => format!("[{}; {n}]", types.qualified(ty)),
            Self::Values(n) => format!("[{}; {n}]", types.rust("Value")),
        }
    }
}

pub(crate) fn field_type(
    records: &[Decl],
    source: &str,
    node: Node,
) -> Result<PropertyType, Error> {
    if let Kind::Call(kind, args) = &node.kind
        && kind == "values"
    {
        if let [
            Node {
                kind: Kind::Number(n),
                ..
            },
        ] = args.as_slice()
            && (1..=255).contains(n)
        {
            return Ok(PropertyType::Values(*n as usize));
        }
        return Err(Error::at(
            source,
            node.offset,
            "operand group size must be in 1..=255",
        ));
    }
    if let Kind::Call(kind, args) = &node.kind
        && kind == "array"
    {
        let [
            element,
            Node {
                kind: Kind::Number(n),
                ..
            },
        ] = args.as_slice()
        else {
            return Err(Error::at(
                source,
                node.offset,
                "array requires an element type and length",
            ));
        };
        if *n > 255 {
            return Err(Error::at(source, node.offset, "array length exceeds 255"));
        }
        let PropertyType::Named(element) = field_type(records, source, element.clone())? else {
            return Err(Error::at(
                source,
                node.offset,
                "array requires a named element type",
            ));
        };
        return Ok(PropertyType::Array(element, *n as usize));
    }
    let sequence = matches!(&node.kind, Kind::Call(name, _) if name == "sequence");
    let optional = matches!(&node.kind, Kind::Call(name, _) if name == "optional");
    let inner = if optional || sequence {
        let Kind::Call(_, ref args) = node.kind else {
            unreachable!()
        };
        if args.len() != 1 {
            return Err(Error::at(
                source,
                node.offset,
                "optional/sequence requires one named type",
            ));
        }
        &args[0]
    } else {
        &node
    };
    let Kind::Name(ty) = &inner.kind else {
        return Err(Error::at(source, node.offset, "expected data type name"));
    };
    if !primitive(ty)
        && !records.iter().any(|r| {
            r.name == *ty
                && (rust_binding(r).is_some() || matches!(&r.kind, DeclKind::Fields(kind) if matches!(kind.as_str(), "struct" | "enum" | "encoding")))
        })
    {
        return Err(Error::at(
            source,
            inner.offset,
            format!("unknown data type `{ty}`"),
        ));
    }
    Ok(if sequence {
        PropertyType::Sequence(ty.clone())
    } else if optional {
        PropertyType::Optional(ty.clone())
    } else {
        PropertyType::Named(ty.clone())
    })
}

pub(crate) fn compile(
    records: &[Decl],
    source: &str,
    rust: &RustTypes,
) -> Result<Vec<RecordDef>, Error> {
    let mut result = Vec::new();
    let mut names = BTreeSet::new();
    for record in records
        .iter()
        .filter(|r| matches!(&r.kind, DeclKind::Fields(kind) if kind == "struct"))
    {
        let fail = |msg: &str| Error::at(source, record.offset, msg);
        model::identifier(source, record.offset, &record.name)?;
        if !names.insert(&record.name) {
            return Err(fail("duplicate property record"));
        }
        // The syntax map supports duplicate detection; source offsets retain declaration order.
        let mut fields = record.fields.iter().collect::<Vec<_>>();
        fields.sort_by_key(|(_, node)| node.offset);
        let members = fields
            .into_iter()
            .map(|(name, node)| {
                model::identifier(source, node.offset, name)?;
                let logical = matches!(&node.kind, Kind::Call(name, _) if rust.policy(name).references.is_operand()).then(|| node.clone());
                let storage = if let Some(Node { kind: Kind::Call(name, _), .. }) = &logical {
                    Node { offset: node.offset, kind: Kind::Name(name.clone()) }
                } else { node.clone() };
                let ty = field_type(records, source, storage)?;
                Ok(RecordField {
                    name: name.clone(),
                    logical,
                    rust: ty.rust(rust),
                    policy: match &ty {
                        PropertyType::Named(name) | PropertyType::Optional(name) | PropertyType::Sequence(name) | PropertyType::Array(name, _) => {
                            rust.policy(name)
                        }
                        PropertyType::Values(_) => Policy {
                            borrowed: false,
                            references: References::Operand,
                            storage: Placement::Auto,
                        },
                    },
                    ty,
                })
            })
            .collect::<Result<_, Error>>()?;
        result.push(RecordDef {
            name: record.name.clone(),
            fields: members,
        });
    }
    Ok(result)
}

pub(crate) fn generate(records: &[RecordDef]) -> String {
    let mut out = String::new();
    for record in records {
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]\npub struct {} {{",
            record.name
        )
        .unwrap();
        for field in &record.fields {
            let ty = &field.rust;
            writeln!(out, "pub {}: {ty},", field.name).unwrap();
        }
        out.push_str("}\n");
    }
    out
}

/// The binding boundary accepts qualified paths, never embedded Rust expressions.

#[cfg(test)]
mod tests {
    use super::*;

    const RECORDS: &str = r#"
        struct PtrIndexImm {
            offset: i32,
            scale: u32,
        }
        struct VectorExtData {
            mask: Value,
            evl: optional(Value),
        }
        struct VectorMemOptions {
            offset: i32,
            flags: MemFlags,
            scale: u8,
            mask: optional(Value),
            evl: optional(Value),
        }
    "#;

    fn checked(source: &str) -> Result<Vec<RecordDef>, Error> {
        {
            let source = format!(
                "type Value = rust(\"crate::Value\");\nencoding MemFlags {{ fields = [volatile(1)]; storage = u16; }}\n{source}"
            );
            crate::model::data::Types::compile(&crate::syntax::parse(&source)?, &source)
                .map(|types| types.records)
        }
    }

    fn rejected(source: &str, message: &str) {
        let error = checked(source).unwrap_err();
        assert!(error.message.contains(message), "{error}");
    }

    #[test]
    fn records_are_definition_owned_and_can_contain_operand_groups() {
        assert_eq!(checked(RECORDS).unwrap().len(), 3);
        assert!(checked(&RECORDS.replace("struct PtrIndexImm", "struct Other")).is_ok());
        assert!(
            checked(&RECORDS.replace("mask: Value", "mask: Value, passthrough: Value")).is_ok()
        );
        rejected(
            &RECORDS.replace("mask: Value", "mask: Value, mask: Value"),
            "duplicate field",
        );
    }

    #[test]
    fn field_order_follows_declarations_not_names() {
        let source = "struct Pair { z: u32, a: i32 }";
        let records = checked(source).unwrap();
        assert_eq!(records[0].fields[0].name, "z");
        let code = generate(&records);
        assert!(code.find("pub z:").unwrap() < code.find("pub a:").unwrap());
        assert!(!code.contains("impl Default"));
    }
}
