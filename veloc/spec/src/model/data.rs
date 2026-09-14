//! Checked declaration values. These are build-time data, not runtime attributes.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::Error;
use crate::model::encoding::Encodings;
use crate::model::records::{PropertyType, RecordDef};
use crate::model::{Fields, identifier, list};
use crate::syntax::{Decl, DeclKind, Kind, Node};

pub(crate) fn fits_number(ty: &str, n: i128) -> bool {
    match ty {
        "u8" => u8::try_from(n).is_ok(),
        "i32" => i32::try_from(n).is_ok(),
        "u32" => u32::try_from(n).is_ok(),
        "u64" => u64::try_from(n).is_ok(),
        "i64" => i64::try_from(n).is_ok(),
        "i128" => true,
        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Value {
    Number(i128),
    Bool(bool),
    Record(String, BTreeMap<String, Value>),
    Variant(String, String, Vec<Value>),
    None,
    Some(Box<Value>),
    Empty(String),
}

impl Value {
    pub fn rust(&self, prefix: &str) -> String {
        match self {
            Self::Number(n) => n.to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Record(ty, fields) => format!(
                "{prefix}{ty} {{ {} }}",
                fields
                    .iter()
                    .map(|(name, value)| format!("{name}: {}", value.rust(prefix)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Variant(ty, variant, args) => {
                let path = format!("{prefix}{ty}::{variant}");
                if args.is_empty() {
                    path
                } else {
                    format!(
                        "{path}({})",
                        args.iter()
                            .map(|v| v.rust(prefix))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Self::None => "None".into(),
            Self::Some(value) => format!("Some({})", value.rust(prefix)),
            Self::Empty(ty) => format!("{prefix}{ty}::empty()"),
        }
    }
}

#[derive(Debug)]
pub(crate) struct EnumDef {
    pub name: String,
    pub variants: Vec<(String, Vec<PropertyType>)>,
}

#[derive(Debug, Default)]
pub(crate) struct Types {
    pub rust: super::records::RustTypes,
    pub names: BTreeSet<String>,
    pub records: Vec<RecordDef>,
    pub enums: Vec<EnumDef>,
}

impl Types {
    pub fn compile(declarations: &[Decl], source: &str) -> Result<Self, Error> {
        let rust = super::records::RustTypes::compile(declarations, source)?;
        let records = crate::model::records::compile(declarations, source, &rust)?;
        let mut enums = Vec::new();
        let mut names = BTreeSet::new();
        for decl in declarations.iter().filter(|d| {
            matches!(&d.kind, DeclKind::Fields(kind) if matches!(kind.as_str(), "struct" | "enum" | "encoding"))
                && !(matches!(&d.kind, DeclKind::Fields(kind) if kind == "encoding") && d.name == "Type")
                || super::records::rust_binding(d).is_some()
        }) {
            if !names.insert(decl.name.clone()) {
                return Err(Error::at(
                    source,
                    decl.offset,
                    format!("duplicate data type `{}`", decl.name),
                ));
            }
        }
        for decl in declarations
            .iter()
            .filter(|d| matches!(&d.kind, DeclKind::Fields(kind) if kind == "enum"))
        {
            let mut fields = Fields::new(source, decl.clone());
            let mut variants = Vec::new();
            let mut seen = BTreeSet::new();
            for node in list(source, fields.take("variants")?)? {
                let offset = node.offset;
                let (variant, args) = match node.kind {
                    Kind::Name(name) => (name, Vec::new()),
                    Kind::Call(name, args) => (name, args),
                    _ => {
                        return Err(Error::at(
                            source,
                            offset,
                            "expected enum variant or Variant(types)",
                        ));
                    }
                };
                identifier(source, offset, &variant)?;
                if !seen.insert(variant.clone()) {
                    return Err(Error::at(source, offset, "duplicate enum variant"));
                }
                let args = args
                    .into_iter()
                    .map(|n| crate::model::records::field_type(declarations, source, n))
                    .collect::<Result<_, _>>()?;
                variants.push((variant, args));
            }
            if variants.is_empty() {
                return Err(fields.error("enum must have at least one variant"));
            }
            fields.finish()?;
            enums.push(EnumDef {
                name: decl.name.clone(),
                variants,
            });
        }
        let types = Self {
            rust,
            names,
            records,
            enums,
        };
        // All declared records/enums are inline values, so recursive layouts are illegal.
        for name in &types.names {
            types.check_cycle(source, name, &mut BTreeSet::new(), &mut BTreeSet::new())?;
        }
        Ok(types)
    }

    fn check_cycle<'a>(
        &'a self,
        source: &str,
        name: &'a str,
        active: &mut BTreeSet<&'a str>,
        done: &mut BTreeSet<&'a str>,
    ) -> Result<(), Error> {
        if done.contains(name) {
            return Ok(());
        }
        if !active.insert(name) {
            return Err(Error::at(
                source,
                0,
                format!("recursive inline data type `{name}`"),
            ));
        }
        let fields: Vec<&PropertyType> =
            if let Some(record) = self.records.iter().find(|r| r.name == name) {
                record.fields.iter().map(|f| &f.ty).collect()
            } else if let Some(en) = self.enums.iter().find(|e| e.name == name) {
                en.variants.iter().flat_map(|(_, args)| args).collect()
            } else {
                Vec::new()
            };
        for ty in fields {
            let name = match ty {
                PropertyType::Named(name)
                | PropertyType::Optional(name)
                | PropertyType::Sequence(name)
                | PropertyType::Array(name, _) => name.as_str(),
                PropertyType::Values(_) => "Value",
            };
            self.check_cycle(source, name, active, done)?;
        }
        active.remove(name);
        done.insert(name);
        Ok(())
    }

    /// Records and enums are acyclic after declaration checking.
    pub fn contains_value(&self, ty: &str) -> bool {
        if !self.rust.policy(ty).references.is_data() {
            return true;
        }
        let has = |ty: &PropertyType| {
            let ty = match ty {
                PropertyType::Named(ty)
                | PropertyType::Optional(ty)
                | PropertyType::Sequence(ty)
                | PropertyType::Array(ty, _) => ty.as_str(),
                PropertyType::Values(_) => "Value",
            };
            self.contains_value(ty)
        };
        self.records
            .iter()
            .find(|r| r.name == ty)
            .is_some_and(|r| r.fields.iter().any(|f| has(&f.ty)))
            || self
                .enums
                .iter()
                .find(|e| e.name == ty)
                .is_some_and(|e| e.variants.iter().any(|(_, args)| args.iter().any(has)))
    }

    pub fn value(
        &self,
        source: &str,
        ty: &PropertyType,
        node: Node,
        encodings: &Encodings,
    ) -> Result<Value, Error> {
        let fail = || {
            Error::at(
                source,
                node.offset,
                format!("expected value of type {}", ty.rust(&self.rust)),
            )
        };
        if let PropertyType::Optional(inner) = ty {
            return match node.kind {
                Kind::Name(ref name) if name == "none" => Ok(Value::None),
                Kind::Call(ref name, ref args) if name == "some" && args.len() == 1 => {
                    Ok(Value::Some(Box::new(self.value(
                        source,
                        &PropertyType::Named(inner.clone()),
                        args[0].clone(),
                        encodings,
                    )?)))
                }
                _ => Err(fail()),
            };
        }
        let PropertyType::Named(ty) = ty else {
            return Err(fail());
        };
        if let Some(record) = self.records.iter().find(|r| r.name == *ty) {
            let Kind::Object(ref name, ref fields) = node.kind else {
                return Err(fail());
            };
            if name != ty {
                return Err(fail());
            }
            for key in fields.keys() {
                if !record.fields.iter().any(|f| f.name == *key) {
                    return Err(Error::at(
                        source,
                        fields[key].offset,
                        format!("unknown field `{key}` in {ty}"),
                    ));
                }
            }
            let mut values = BTreeMap::new();
            for field in &record.fields {
                let input = fields.get(&field.name).ok_or_else(|| {
                    Error::at(
                        source,
                        node.offset,
                        format!("missing field `{}` in {ty}", field.name),
                    )
                })?;
                let value = self.value(source, &field.ty, input.clone(), encodings)?;
                values.insert(field.name.clone(), value);
            }
            return Ok(Value::Record(ty.clone(), values));
        }
        if let Some(en) = self.enums.iter().find(|e| e.name == *ty) {
            let (variant, args) = match &node.kind {
                Kind::Name(variant) => (variant.clone(), Vec::new()),
                Kind::Call(variant, args) => (variant.clone(), args.clone()),
                _ => return Err(fail()),
            };
            let Some((_, params)) = en.variants.iter().find(|(name, _)| *name == variant) else {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown {ty} variant `{variant}`"),
                ));
            };
            if params.len() != args.len() {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("{ty}::{variant} requires {} arguments", params.len()),
                ));
            }
            let values = params
                .iter()
                .zip(args)
                .map(|(ty, arg)| self.value(source, ty, arg, encodings))
                .collect::<Result<_, _>>()?;
            return Ok(Value::Variant(ty.clone(), variant, values));
        }
        match &node.kind {
            Kind::Number(n) if fits_number(ty, i128::from(*n)) => Ok(Value::Number(i128::from(*n))),
            Kind::Integer(n) if fits_number(ty, *n) => Ok(Value::Number(*n)),
            Kind::Name(n) if ty == "bool" && matches!(n.as_str(), "true" | "false") => {
                Ok(Value::Bool(n == "true"))
            }
            Kind::Name(n) if n == "empty" && encodings.contains_key(ty) => {
                Ok(Value::Empty(ty.clone()))
            }
            _ => Err(fail()),
        }
    }

    fn emitted_records(&self, layouts: &[String], metadata: Option<&str>) -> Vec<RecordDef> {
        // Storage is a use of a record, not its Rust data representation. A record
        // also referenced as a field, enum payload or metadata still needs a struct.
        let referenced = self
            .records
            .iter()
            .flat_map(|r| r.fields.iter().map(|f| &f.ty))
            .chain(
                self.enums
                    .iter()
                    .flat_map(|e| e.variants.iter().flat_map(|(_, args)| args)),
            )
            .filter_map(|ty| match ty {
                PropertyType::Named(name)
                | PropertyType::Optional(name)
                | PropertyType::Sequence(name)
                | PropertyType::Array(name, _) => Some(name.as_str()),
                PropertyType::Values(_) => None,
            })
            .chain(metadata)
            .collect::<BTreeSet<_>>();
        self.records
            .iter()
            .filter(|r| !layouts.contains(&r.name) || referenced.contains(r.name.as_str()))
            .cloned()
            .collect()
    }

    pub(crate) fn validate_sequences(
        &self,
        layouts: &[String],
        metadata: Option<&str>,
        source: &str,
    ) -> Result<(), Error> {
        // Borrowed sequences are currently storage views, not standalone data.
        // Reject unsupported lifetime propagation before emitting Rust.
        let fields = self.emitted_records(layouts, metadata);
        if fields
            .iter()
            .flat_map(|r| &r.fields)
            .any(|f| matches!(f.ty, PropertyType::Sequence(_)))
            || self
                .enums
                .iter()
                .flat_map(|e| &e.variants)
                .flat_map(|(_, args)| args)
                .any(|ty| matches!(ty, PropertyType::Sequence(_)))
        {
            return Err(Error::at(
                source,
                0,
                "sequence fields are supported only in standalone operand storage layouts",
            ));
        }
        Ok(())
    }

    pub fn generate(&self, layouts: &[String], metadata: Option<&str>) -> String {
        let mut out = crate::model::records::generate(&self.emitted_records(layouts, metadata));
        for en in &self.enums {
            let traits = if layouts.contains(&en.name) {
                "Debug, Clone, Copy"
            } else {
                "Debug, Clone, Copy, PartialEq, Eq, Hash"
            };
            writeln!(out, "#[derive({traits})]\npub enum {} {{", en.name).unwrap();
            for (name, args) in &en.variants {
                if args.is_empty() {
                    writeln!(out, "{name},").unwrap();
                } else {
                    writeln!(
                        out,
                        "{name}({}),",
                        args.iter()
                            .map(|ty| ty.rust(&self.rust))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                    .unwrap();
                }
            }
            out.push_str("}\n");
        }
        out
    }
}
