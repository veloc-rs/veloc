//! Structured logical properties and their generated Rust representation.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::syntax::{Kind, Node, Record};
use crate::{Error, model};

/// Rust type bindings are nominal in defs; paths only control code emission.
#[derive(Debug, Clone, Default)]
pub(crate) struct RustTypes {
    external: BTreeMap<String, String>,
}

pub(crate) fn rust_binding(record: &Record) -> Option<&Node> {
    (record.kind == "type")
        .then(|| record.fields.get("expr"))
        .flatten()
        .filter(|node| matches!(&node.kind, Kind::Call(name, _) if name == "rust"))
}

pub(crate) fn primitive(name: &str) -> bool {
    matches!(
        name,
        "bool" | "u8" | "u32" | "u64" | "i32" | "i64" | "i128" | "f64"
    )
}

impl RustTypes {
    pub fn compile(records: &[Record], source: &str) -> Result<Self, Error> {
        let mut result = Self::default();
        for record in records {
            let Some(node) = rust_binding(record) else {
                continue;
            };
            if primitive(&record.name)
                || matches!(
                    record.name.as_str(),
                    "ValueList" | "BlockCall" | "JumpTable"
                )
            {
                return Err(Error::at(
                    source,
                    record.offset,
                    "Rust type binding conflicts with a built-in type",
                ));
            }
            let Kind::Call(_, args) = &node.kind else {
                unreachable!()
            };
            let [
                Node {
                    kind: Kind::Text(path),
                    ..
                },
            ] = args.as_slice()
            else {
                return Err(Error::at(
                    source,
                    node.offset,
                    "rust requires one type path string",
                ));
            };
            rust_path(source, node.offset, path)?;
            if result
                .external
                .insert(record.name.clone(), path.clone())
                .is_some()
            {
                return Err(Error::at(
                    source,
                    record.offset,
                    "duplicate Rust type binding",
                ));
            }
        }
        Ok(result)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.external.contains_key(name)
    }

    pub fn rust(&self, name: &str) -> String {
        self.external
            .get(name)
            .cloned()
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PropertyType {
    Named(String),
    Optional(String),
    Values(usize),
}

impl PropertyType {
    pub fn rust(&self, types: &RustTypes) -> String {
        match self {
            Self::Named(ty) => types.rust(ty),
            Self::Optional(ty) => format!("Option<{}>", types.rust(ty)),
            Self::Values(n) => format!("[{}; {n}]", types.rust("Value")),
        }
    }
}

pub(crate) fn field_type(
    records: &[Record],
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
    let optional = matches!(&node.kind, Kind::Call(name, _) if name == "optional");
    let inner = if optional {
        let Kind::Call(_, ref args) = node.kind else {
            unreachable!()
        };
        if args.len() != 1 {
            return Err(Error::at(source, node.offset, "optional requires one type"));
        }
        &args[0]
    } else {
        &node
    };
    let Kind::Name(ty) = &inner.kind else {
        return Err(Error::at(source, node.offset, "expected data type name"));
    };
    if !primitive(ty)
        && !matches!(ty.as_str(), "ValueList" | "BlockCall" | "JumpTable")
        && !crate::storage::operands::is_role(ty)
        && !records.iter().any(|r| {
            r.name == *ty
                && (rust_binding(r).is_some()
                    || matches!(
                        r.kind.as_str(),
                        "struct" | "enum" | "flags" | "encoding" | "comparison"
                    ))
        })
    {
        return Err(Error::at(
            source,
            inner.offset,
            format!("unknown data type `{ty}`"),
        ));
    }
    Ok(if optional {
        PropertyType::Optional(ty.clone())
    } else {
        PropertyType::Named(ty.clone())
    })
}

pub(crate) fn compile(
    records: &[Record],
    source: &str,
    rust: &RustTypes,
) -> Result<Vec<RecordDef>, Error> {
    let mut result = Vec::new();
    let mut names = BTreeSet::new();
    for record in records.iter().filter(|r| r.kind == "struct") {
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
                let ty = field_type(records, source, node.clone())?;
                Ok(RecordField {
                    name: name.clone(),
                    rust: ty.rust(rust),
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
pub(crate) fn rust_path(source: &str, offset: usize, path: &str) -> Result<(), Error> {
    // Accept paths, not arbitrary Rust code, references or generic types.
    let mut parts = path.split("::");
    let first = parts.next().unwrap_or_default();
    if first.is_empty() {
        return Err(Error::at(
            source,
            offset,
            "Rust type path must be nonempty and qualified",
        ));
    }
    if !matches!(first, "crate") {
        model::identifier(source, offset, first)?;
    }
    let rest = parts.collect::<Vec<_>>();
    if rest.is_empty() {
        return Err(Error::at(
            source,
            offset,
            "Rust type path must be qualified",
        ));
    }
    for part in rest {
        model::identifier(source, offset, part)?;
    }

    Ok(())
}

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
                "type Value = rust(\"crate::Value\");\nencoding MemFlags {{ fields: [volatile(1)], storage: u16 }}\n{source}"
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
