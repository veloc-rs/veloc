//! Checked text projections over logical parameters, not storage layouts.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::model::data::Value;
use crate::model::records::{PropertyType, RecordDef};
use crate::model::{Op, ParamKind, Pattern, TypeDef, TypeList};
use crate::syntax::Kind;

mod template;

#[derive(Debug)]
pub(super) struct Schema {
    pub args: Vec<Item>,
    pub named: Vec<Named>,
    pub flags: Option<String>,
    pub bindings: Vec<(String, Value)>,
}

#[derive(Debug)]
pub(super) enum Item {
    Atom(Atom),
    Space(Box<Item>, Box<Item>),
    Invoke {
        callee: Atom,
        args: Atom,
        signature: CallSignature,
    },
}

#[derive(Debug)]
pub(super) enum CallSignature {
    Value,
    Field(Atom),
    Function,
}

#[derive(Debug, Clone)]
pub(super) struct Atom {
    pub path: String,
    pub kind: AtomKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AtomKind {
    Value,
    Values,
    Successor,
    Successors,
    Scalar(String),
    OptionalValue,
    Integer,
    Bytes,
}

#[derive(Debug)]
pub(super) struct Named {
    pub atom: Atom,
    pub key: String,
    pub mode: Mode,
}

#[derive(Debug)]
pub(super) enum Mode {
    Required,
    Optional,
}

struct Checker<'a> {
    source: &'a str,
    leaves: BTreeMap<String, AtomKind>,
    used: BTreeSet<String>,
    typed: BTreeSet<String>,
}

pub(super) fn compile(
    op: &Op,
    records: &[RecordDef],
    source: &str,
    types: &crate::types::Types,
) -> Result<Schema, Error> {
    let mut checker = Checker {
        source,
        leaves: BTreeMap::new(),
        used: BTreeSet::new(),
        typed: BTreeSet::new(),
    };
    for param in &op.params {
        if let ParamKind::Property(ty) = &param.kind
            && let Some(record) = records.iter().find(|record| record.name == *ty)
        {
            for field in &record.fields {
                let kind = match &field.ty {
                    PropertyType::Named(_) if field.policy.references.is_operand() => {
                        AtomKind::Value
                    }
                    PropertyType::Named(ty) => AtomKind::Scalar(ty.clone()),
                    PropertyType::Optional(_) if field.policy.references.is_operand() => {
                        AtomKind::OptionalValue
                    }
                    _ => return Err(checker.error(op.offset, "unsupported optional property")),
                };
                checker
                    .leaves
                    .insert(format!("{}.{}", param.name, field.name), kind);
            }
            continue;
        }
        let kind = match &param.kind {
            ParamKind::Value => AtomKind::Value,
            ParamKind::Values => AtomKind::Values,
            ParamKind::Successor => AtomKind::Successor,
            ParamKind::Successors => AtomKind::Successors,
            ParamKind::Property(ty) => AtomKind::Scalar(ty.clone()),
        };
        checker.leaves.insert(param.name.clone(), kind);
    }

    let mut schema = Schema {
        args: Vec::new(),
        named: Vec::new(),
        flags: None,
        bindings: Vec::new(),
    };
    if let Some(node) = &op.text {
        let Kind::Text(text) = &node.kind else {
            return Err(checker.error(node.offset, "expected a quoted text template"));
        };
        template::compile(&mut checker, &mut schema, text, node.offset)?;
    } else {
        for param in &op.params {
            schema
                .args
                .push(Item::Atom(checker.atom(&param.name, None, op.offset)?));
        }
    }

    if schema.args.iter().any(|item| {
        matches!(
            item,
            Item::Atom(Atom {
                kind: AtomKind::Values,
                ..
            })
        )
    }) && schema.args.len() != 1
    {
        return Err(checker.error(
            op.offset,
            "top-level values must be the only positional item",
        ));
    }
    for ty in &checker.typed {
        let description = match ty.as_str() {
            "Float" => "scalar float",
            "Int" => "scalar integer",
            "VectorConst" => "vector",
            _ => unreachable!("typed text property"),
        };
        let allowed = types.property_types(ty).expect("typed text property");
        if !result_in_set(&op.signature, types, &allowed) {
            return Err(checker.error(
                op.offset,
                format!(
                    "{} text atoms require a {description} first result",
                    description.strip_prefix("scalar ").unwrap_or(description)
                ),
            ));
        }
    }
    for path in checker.leaves.keys() {
        if !checker.used.contains(path) {
            return Err(Error::at(
                source,
                op.offset,
                format!("text projection does not consume `{path}`"),
            ));
        }
    }
    Ok(schema)
}

impl Checker<'_> {
    fn error(&self, offset: usize, message: impl Into<String>) -> Error {
        Error::at(self.source, offset, message)
    }

    fn consume(&mut self, path: &str, offset: usize) -> Result<AtomKind, Error> {
        let leaf = self.leaves.get(path).ok_or_else(|| {
            self.error(
                offset,
                format!("unknown text field `{path}`; records require a field path"),
            )
        })?;
        if !self.used.insert(path.into()) {
            return Err(self.error(
                offset,
                format!("text field `{path}` is consumed more than once"),
            ));
        }
        Ok(leaf.clone())
    }

    fn atom(&mut self, path: &str, codec: Option<&str>, offset: usize) -> Result<Atom, Error> {
        let kind = self.consume(path, offset)?;
        let kind = match (codec, kind) {
            (Some("integer"), AtomKind::Scalar(ty)) if ty == "u64" => AtomKind::Integer,
            (None, AtomKind::Scalar(ty))
                if matches!(ty.as_str(), "Float" | "Int" | "VectorConst") =>
            {
                self.typed.insert(ty.clone());
                AtomKind::Scalar(ty)
            }
            (Some("bytes"), AtomKind::Scalar(ty)) if ty == "Bytes" => AtomKind::Bytes,
            (Some(codec), _) => {
                return Err(self.error(
                    offset,
                    format!("text codec `{codec}` is incompatible with `{path}`"),
                ));
            }
            (None, AtomKind::Scalar(ty)) if !simple_scalar(&ty) => {
                return Err(self.error(
                    offset,
                    format!("property `{path}` of type `{ty}` needs an explicit text projection"),
                ));
            }
            (None, kind) => kind,
        };
        Ok(Atom {
            path: path.into(),
            kind,
        })
    }
}

fn simple_scalar(ty: &str) -> bool {
    matches!(
        ty,
        "u8" | "u32"
            | "u64"
            | "i32"
            | "bool"
            | "FuncId"
            | "SigId"
            | "Intrinsic"
            | "IntCC"
            | "FloatCC"
            | "Int"
    )
}

fn result_in_set(
    signature: &TypeDef,
    types: &crate::types::Types,
    allowed: &crate::types::TypeSet,
) -> bool {
    let Some(result) = signature
        .results
        .patterns()
        .and_then(|results| results.first())
    else {
        return false;
    };
    let accepts = |set: &crate::types::TypeSet| set.subset_of(allowed);
    match result {
        Pattern::Exact(ty) => accepts(&types.exact[ty]),
        Pattern::Set(set) | Pattern::Bind(_, set) | Pattern::Property(_, set) => accepts(set),
        Pattern::Same(slot) => {
            let operands = match &signature.operands {
                TypeList::Fixed(operands) | TypeList::Variadic(operands) => operands.as_slice(),
                TypeList::Signature => &[],
            };
            operands.iter().any(|pattern| {
                matches!(pattern, Pattern::Bind(other, set) if slot == other && accepts(set))
            })
        }
        _ => false,
    }
}

fn single_token(kind: &AtomKind) -> bool {
    matches!(kind, AtomKind::Value | AtomKind::Integer | AtomKind::Bytes)
        || matches!(kind, AtomKind::Scalar(ty) if !matches!(ty.as_str(), "SigId" | "FuncId" | "VectorConst"))
}
