//! Structured logical properties and their generated Rust representation.
use std::collections::BTreeSet;
use std::fmt::Write;

use crate::syntax::{Kind, Node, Record};
use crate::{Error, model};

#[derive(Debug, Clone)]
pub(crate) struct RecordDef {
    pub name: String,
    pub fields: Vec<RecordField>,
}

#[derive(Debug, Clone)]
pub(crate) struct RecordField {
    pub name: String,
    pub ty: PropertyType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PropertyType {
    Named(String),
    Optional(String),
    Values(usize),
}

impl PropertyType {
    pub fn rust(&self) -> String {
        match self {
            Self::Named(ty) => ty.clone(),
            Self::Optional(ty) => format!("Option<{ty}>"),
            Self::Values(n) => format!("[Value; {n}]"),
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
    if !matches!(
        ty.as_str(),
        "i32"
            | "i64"
            | "f64"
            | "u32"
            | "u64"
            | "u8"
            | "bool"
            | "Value"
            | "ValueList"
            | "BlockCall"
            | "JumpTable"
            | "FuncId"
            | "SigId"
            | "ConstantPoolId"
            | "Intrinsic"
            | "IntCC"
            | "FloatCC"
            | "Int"
            | "Float"
            | "VectorConst"
            | "SymbolId"
    ) && !crate::storage::operands::is_role(ty)
        && !records.iter().any(|r| {
            r.name == *ty && matches!(r.kind.as_str(), "record" | "enum" | "flags" | "encoding")
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

pub(crate) fn compile(records: &[Record], source: &str) -> Result<Vec<RecordDef>, Error> {
    let mut result = Vec::new();
    let mut names = BTreeSet::new();
    for record in records.iter().filter(|r| r.kind == "record") {
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
                Ok(RecordField {
                    name: name.clone(),
                    ty: field_type(records, source, node.clone())?,
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
            let ty = field.ty.rust();
            writeln!(out, "pub {}: {ty},", field.name).unwrap();
        }
        out.push_str("}\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORDS: &str = r#"
        record PtrIndexImm {
            offset: i32,
            scale: u32,
        }
        record VectorExtData {
            mask: Value,
            evl: optional(Value),
        }
        record VectorMemOptions {
            offset: i32,
            flags: MemFlags,
            scale: u8,
            mask: optional(Value),
            evl: optional(Value),
        }
    "#;

    fn checked(source: &str) -> Result<Vec<RecordDef>, Error> {
        {
            let source =
                format!("encoding MemFlags {{ fields: [volatile(1)], storage: u16 }}\n{source}");
            crate::data::Types::compile(&crate::syntax::parse(&source)?, &source)
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
        assert!(checked(&RECORDS.replace("record PtrIndexImm", "record Other")).is_ok());
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
        let source = "record Pair { z: u32, a: i32 }";
        let records = checked(source).unwrap();
        assert_eq!(records[0].fields[0].name, "z");
        let code = generate(&records);
        assert!(code.find("pub z:").unwrap() < code.find("pub a:").unwrap());
        assert!(!code.contains("impl Default"));
    }
}
