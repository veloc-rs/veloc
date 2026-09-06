//! Structured logical properties and their generated Rust representation.
use std::collections::BTreeSet;
use std::fmt::Write;

use crate::syntax::{Kind, Record};
use crate::{Error, model};

#[derive(Debug, Clone)]
pub(crate) struct RecordDef {
    pub name: String,
    pub storage: String,
    pub fields: Vec<RecordField>,
}

#[derive(Debug, Clone)]
pub(crate) struct RecordField {
    pub name: String,
    pub ty: PropertyType,
    pub default: Option<DefaultValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PropertyType {
    Named(String),
    Optional(String),
}

#[derive(Debug, Clone)]
pub(crate) enum DefaultValue {
    Number(u32),
    None,
    Empty,
}

impl DefaultValue {
    pub(crate) fn rust(&self) -> String {
        match self {
            Self::Number(n) => n.to_string(),
            Self::None => "None".into(),
            Self::Empty => "crate::MemFlags::empty()".into(),
        }
    }
}

pub(crate) fn compile(records: &[Record], source: &str) -> Result<Vec<RecordDef>, Error> {
    let mut result = Vec::new();
    let mut names = BTreeSet::new();
    let mut stores = BTreeSet::new();
    for record in records.iter().filter(|r| r.kind == "record") {
        let fail = |msg: &str| Error::at(source, record.offset, msg);
        model::identifier(source, record.offset, &record.name)?;
        if !names.insert(&record.name) {
            return Err(fail("duplicate property record"));
        }
        for key in record.fields.keys() {
            if !matches!(key.as_str(), "storage" | "fields") {
                return Err(fail("unknown property record field"));
            }
        }
        let Some(storage) = record.fields.get("storage") else {
            return Err(fail("record has no storage type"));
        };
        let Kind::Name(storage) = &storage.kind else {
            return Err(fail("expected storage type name"));
        };
        if !matches!(
            storage.as_str(),
            "PtrIndexImmId" | "VectorMemExtId" | "VectorExtId"
        ) {
            return Err(fail("unsupported property pool storage type"));
        }
        if !stores.insert(storage.clone()) {
            return Err(fail("duplicate property pool storage type"));
        }
        let Some(fields) = record.fields.get("fields") else {
            return Err(fail("record has no fields"));
        };
        let Kind::List(fields) = &fields.kind else {
            return Err(fail("expected record field list"));
        };
        let mut members = Vec::new();
        let mut seen = BTreeSet::new();
        for field in fields {
            let fail = |msg: &str| Error::at(source, field.offset, msg);
            let Kind::Call(name, args) = &field.kind else {
                return Err(fail("expected field(type, default?)"));
            };
            model::identifier(source, field.offset, name)?;
            if !seen.insert(name.clone()) {
                return Err(fail("duplicate property field"));
            }
            if !(1..=2).contains(&args.len()) {
                return Err(fail("expected field(type, default?)"));
            }
            let ty = match &args[0].kind {
                Kind::Name(ty)
                    if matches!(
                        ty.as_str(),
                        "i32" | "u32" | "u64" | "u8" | "bool" | "Value" | "MemFlags"
                    ) =>
                {
                    PropertyType::Named(ty.clone())
                }
                Kind::Call(kind, args) if kind == "optional" && args.len() == 1 => {
                    let Kind::Name(ty) = &args[0].kind else {
                        return Err(fail("expected optional value type"));
                    };
                    if ty != "Value" {
                        return Err(fail("only optional(Value) is supported"));
                    }
                    PropertyType::Optional(ty.clone())
                }
                _ => return Err(fail("unsupported property field type")),
            };
            let default = match args.get(1).map(|n| &n.kind) {
                None => None,
                Some(Kind::Number(n)) if numeric_default(&ty, *n) => Some(DefaultValue::Number(*n)),
                Some(Kind::Name(n)) if n == "none" && matches!(ty, PropertyType::Optional(_)) => {
                    Some(DefaultValue::None)
                }
                Some(Kind::Name(n))
                    if n == "empty" && ty == PropertyType::Named("MemFlags".into()) =>
                {
                    Some(DefaultValue::Empty)
                }
                _ => return Err(fail("default is incompatible with property field type")),
            };
            members.push(RecordField {
                name: name.clone(),
                ty,
                default,
            });
        }
        validate_runtime_contract(source, record.offset, &record.name, storage, &members)?;
        result.push(RecordDef {
            name: record.name.clone(),
            storage: storage.clone(),
            fields: members,
        });
    }
    Ok(result)
}

/// These pools still have typed DFG constructors and operand visitors. Their
/// Rust-facing fields cannot change independently of those runtime adapters.
/// Order and defaults remain definition-owned, but accepting an extra Value
/// here would silently omit it from use-def traversal and replacement.
fn validate_runtime_contract(
    source: &str,
    offset: usize,
    name: &str,
    storage: &str,
    fields: &[RecordField],
) -> Result<(), Error> {
    let (expected_name, expected): (&str, &[(&str, &str)]) = match storage {
        "PtrIndexImmId" => ("PtrIndexImm", &[("offset", "i32"), ("scale", "u32")]),
        "VectorExtId" => (
            "VectorExtData",
            &[("mask", "Value"), ("evl", "optional(Value)")],
        ),
        "VectorMemExtId" => (
            "VectorMemOptions",
            &[
                ("offset", "i32"),
                ("flags", "MemFlags"),
                ("scale", "u8"),
                ("mask", "optional(Value)"),
                ("evl", "optional(Value)"),
            ],
        ),
        _ => unreachable!("storage types were checked"),
    };
    let fail = |message| Error::at(source, offset, message);
    if name != expected_name {
        return Err(fail(format!(
            "pool `{storage}` requires runtime record `{expected_name}`"
        )));
    }
    for &(name, ty) in expected {
        let Some(field) = fields.iter().find(|field| field.name == name) else {
            return Err(fail(format!(
                "runtime pool record `{expected_name}` requires field `{name}`"
            )));
        };
        let actual = match &field.ty {
            PropertyType::Named(ty) => ty.clone(),
            PropertyType::Optional(ty) => format!("optional({ty})"),
        };
        if actual != ty {
            return Err(fail(format!(
                "runtime pool field `{expected_name}.{name}` requires `{ty}`, got `{actual}`"
            )));
        }
    }
    if let Some(field) = fields
        .iter()
        .find(|field| !expected.iter().any(|(name, _)| *name == field.name))
    {
        return Err(fail(format!(
            "unsupported runtime pool field `{expected_name}.{}`; extend the DFG adapter before adding fields",
            field.name
        )));
    }
    Ok(())
}

pub(crate) fn numeric_default(ty: &PropertyType, n: u32) -> bool {
    match ty {
        PropertyType::Named(ty) => match ty.as_str() {
            "u8" => u8::try_from(n).is_ok(),
            "i32" => i32::try_from(n).is_ok(),
            "u32" | "u64" => true,
            _ => false,
        },
        _ => false,
    }
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
            let ty = match &field.ty {
                PropertyType::Named(ty) => ty.clone(),
                PropertyType::Optional(ty) => format!("Option<{ty}>"),
            };
            writeln!(out, "pub {}: {ty},", field.name).unwrap();
        }
        out.push_str("}\n");
        if record.fields.iter().all(|f| f.default.is_some()) {
            writeln!(
                out,
                "impl Default for {} {{ fn default() -> Self {{ Self {{",
                record.name
            )
            .unwrap();
            for field in &record.fields {
                writeln!(
                    out,
                    "{}: {},",
                    field.name,
                    field.default.as_ref().unwrap().rust()
                )
                .unwrap();
            }
            out.push_str("} } }\n");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORDS: &str = r#"
        record PtrIndexImm {
            storage: PtrIndexImmId, fields: [offset(i32, 0), scale(u32, 1)]
        }
        record VectorExtData {
            storage: VectorExtId, fields: [mask(Value), evl(optional(Value), none)]
        }
        record VectorMemOptions {
            storage: VectorMemExtId,
            fields: [offset(i32, 0), flags(MemFlags, empty), scale(u8, 1), mask(optional(Value), none), evl(optional(Value), none)]
        }
    "#;

    fn checked(source: &str) -> Result<Vec<RecordDef>, Error> {
        compile(&crate::syntax::parse(source)?, source)
    }

    fn rejected(source: &str, message: &str) {
        let error = checked(source).unwrap_err();
        assert!(error.message.contains(message), "{error}");
    }

    #[test]
    fn records_keep_runtime_names_and_field_types() {
        assert_eq!(checked(RECORDS).unwrap().len(), 3);
        rejected(
            &RECORDS.replace("record PtrIndexImm", "record Other"),
            "requires runtime record",
        );
        rejected(
            &RECORDS.replace("scale(u32, 1)", "scale(u8, 1)"),
            "PtrIndexImm.scale",
        );
        rejected(
            &RECORDS.replace("mask(Value)", "mask(u32)"),
            "VectorExtData.mask",
        );
        rejected(
            &RECORDS.replace("mask(optional(Value), none)", "mask(Value)"),
            "VectorMemOptions.mask",
        );
        rejected(
            &RECORDS.replace(", scale(u32, 1)", ""),
            "requires field `scale`",
        );
    }

    #[test]
    fn extending_fields_requires_extending_the_runtime_adapter() {
        rejected(
            &RECORDS.replace("mask(Value)", "mask(Value), passthrough(Value)"),
            "extend the DFG adapter",
        );
        rejected(
            &RECORDS.replace("scale(u32, 1)", "scale(u32, 1), extra(u8, 0)"),
            "extend the DFG adapter",
        );
    }

    #[test]
    fn field_order_and_defaults_are_owned_by_the_definition() {
        let source = RECORDS.replace(
            "offset(i32, 0), scale(u32, 1)",
            "scale(u32, 4), offset(i32, 17)",
        );
        let records = checked(&source).unwrap();
        assert_eq!(records[0].fields[0].name, "scale");
        let code = generate(&records);
        assert!(code.contains("scale: 4"));
        assert!(code.contains("offset: 17"));
    }
}
