//! Named flag sets and the MIR trait/memory-effect vocabulary.

use std::collections::{BTreeMap, BTreeSet};

use std::fmt::Write;

use super::encoding::{BitLayout, Storage};
use crate::Error;
use crate::model::{Fields, identifier, list, name};
use crate::syntax::{Kind, Node, Record};

#[derive(Debug)]
pub(crate) struct Flag {
    pub name: String,
    pub bit: u8,
}

#[derive(Debug)]
pub(crate) struct Flags {
    pub storage: Storage,
    pub members: Vec<Flag>,
    pub separator: String,
    all: u128,
}

impl Flags {
    fn compile(source: &str, fields: &mut Fields<'_>) -> Result<Self, Error> {
        let storage = Storage::parse(source, fields.take("storage")?)?;
        let mut layout = BitLayout::new(storage);
        let mut members: Vec<Flag> = Vec::new();
        for node in list(source, fields.take("members")?)? {
            let Kind::Call(name, args) = node.kind else {
                return Err(Error::at(source, node.offset, "expected MEMBER(bit)"));
            };
            identifier(source, node.offset, &name)?;
            if name != name.to_ascii_uppercase()
                || matches!(name.as_str(), "NONE" | "ALL" | "NAMES")
            {
                return Err(Error::at(
                    source,
                    node.offset,
                    "flag names must be uppercase and cannot be NONE, ALL or NAMES",
                ));
            }
            let [bit] = args.as_slice() else {
                return Err(Error::at(
                    source,
                    node.offset,
                    "flag member requires exactly one bit position",
                ));
            };
            let bit = number(source, bit.clone())?;
            layout.insert(source, node.offset, &name, 1, bit)?;
            members.push(Flag {
                name,
                bit: bit as u8,
            });
        }
        let separator = fields.take("separator")?;
        let Kind::Text(separator) = separator.kind else {
            return Err(Error::at(
                source,
                separator.offset,
                "flag separator must be a string",
            ));
        };
        Ok(Self {
            storage,
            members,
            separator,
            all: layout.used,
        })
    }

    pub fn all(&self) -> u128 {
        self.all
    }
}

#[derive(Debug)]
pub(crate) enum Effect {
    Known(Vec<String>),
    Unknown,
}

impl Effect {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::Known(members) if members.is_empty())
    }
}

#[derive(Debug, Default)]
pub(crate) struct Builtins {
    pub encodings: BTreeMap<String, BitLayout>,
    pub flags: BTreeMap<String, Flags>,
}

impl Builtins {
    pub fn compile(records: &[Record], source: &str) -> Result<Self, Error> {
        let mut defs = Self::default();
        for record in records.iter().filter(|r| Self::is_definition(&r.kind)) {
            let mut fields = Fields::new(source, record.clone());
            match record.kind.as_str() {
                "flags" | "encoding" => {
                    if matches!(
                        record.name.as_str(),
                        "Opcode" | "OpFormat" | "TypeClass" | "OpSpec" | "TypeError" | "type_rules"
                    ) || records.iter().any(|other| {
                        (other.kind == "comparison"
                            || (other.kind != record.kind
                                && matches!(other.kind.as_str(), "flags" | "encoding")))
                            && other.name == record.name
                    }) {
                        return Err(fields.error(format!(
                            "bit layout `{}` conflicts with a MIR opcode type or module",
                            record.name
                        )));
                    }
                    if record.kind == "flags" {
                        let flags = Flags::compile(source, &mut fields)?;
                        defs.flags.insert(record.name.clone(), flags);
                    } else {
                        let layout = BitLayout::packed(source, &mut fields)?;
                        layout.check_methods(source, record.offset)?;
                        defs.encodings.insert(record.name.clone(), layout);
                    }
                }
                _ => unreachable!(),
            }
            fields.finish()?;
        }
        Ok(defs)
    }

    pub fn is_definition(kind: &str) -> bool {
        matches!(kind, "flags" | "encoding")
    }

    pub fn members(&self, source: &str, node: Node, set: &str) -> Result<Vec<String>, Error> {
        let offset = node.offset;
        let mut seen = BTreeSet::new();
        for node in list(source, node)? {
            let member = self.reference(source, node, set, |name| self.has_flag(set, name))?;
            if !seen.insert(member) {
                return Err(Error::at(source, offset, format!("duplicate {set} member")));
            }
        }
        // Canonical order comes from the declaration, independent of spelling order.
        Ok(self
            .flags
            .get(set)
            .into_iter()
            .flat_map(|flags| &flags.members)
            .filter(|flag| seen.contains(&flag.name))
            .map(|flag| flag.name.clone())
            .collect())
    }

    fn reference(
        &self,
        source: &str,
        node: Node,
        what: &str,
        contains: impl FnOnce(&str) -> bool,
    ) -> Result<String, Error> {
        let offset = node.offset;
        let name = name(source, node)?;
        if contains(&name) {
            Ok(name)
        } else {
            Err(Error::at(
                source,
                offset,
                format!("unknown {what} `{name}`"),
            ))
        }
    }

    pub fn has_trait(&self, name: &str) -> bool {
        self.has_flag("OpTraits", name)
    }

    fn has_flag(&self, set: &str, name: &str) -> bool {
        self.flags
            .get(set)
            .is_some_and(|flags| flags.members.iter().any(|flag| flag.name == name))
    }
}

fn number(source: &str, node: Node) -> Result<u32, Error> {
    match node.kind {
        Kind::Number(value) => Ok(value),
        _ => Err(Error::at(source, node.offset, "expected a number")),
    }
}

const HEADER: &str = "// @generated by veloc-opgen; edit defs/builtins.ops.\n";

pub(crate) fn contracts(defs: &Builtins) -> String {
    let mut out = HEADER.to_owned();
    for (name, layout) in &defs.encodings {
        out.push_str(&layout.generate(name));
    }
    for (name, set) in &defs.flags {
        flags(&mut out, name, set);
    }
    out
}

/// Emit a const expression for members already checked against a flags declaration.
pub(crate) fn flag_set(ty: &str, members: &[String]) -> String {
    let Some((first, rest)) = members.split_first() else {
        return format!("{ty}::empty()");
    };
    let mut out = format!("{ty}::{first}");
    for member in rest {
        write!(out, ".union({ty}::{member})").unwrap();
    }
    out
}

fn flags(out: &mut String, ty: &str, flags: &Flags) {
    writeln!(
        out,
        "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]\npub struct {ty}({});",
        flags.storage.name()
    )
    .unwrap();
    writeln!(out, "impl {ty} {{").unwrap();
    writeln!(
        out,
        "pub const NONE: Self = Self(0);\npub const ALL: Self = Self({});",
        flags.all()
    )
    .unwrap();
    out.push_str("pub const fn empty() -> Self { Self::NONE }\npub const fn is_empty(self) -> bool { self.0 == 0 }\npub const fn union(self, other: Self) -> Self { Self(self.0 | other.0) }\npub const fn contains(self, other: Self) -> bool { self.0 & other.0 == other.0 }\npub const fn intersects(self, other: Self) -> bool { self.0 & other.0 != 0 }\n");
    for flag in &flags.members {
        writeln!(
            out,
            "pub const {}: Self = Self(1 << {});",
            flag.name, flag.bit
        )
        .unwrap();
    }
    out.push_str("const NAMES: &'static [(Self, &'static str)] = &[\n");
    for flag in &flags.members {
        writeln!(
            out,
            "(Self::{}, {:?}),",
            flag.name,
            flag.name.to_ascii_lowercase().replace('_', "-")
        )
        .unwrap();
    }
    out.push_str("];\n}\n");
    writeln!(out, "impl core::fmt::Display for {ty} {{\nfn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{\nif self.is_empty() {{ return f.write_str(\"none\"); }}\nlet mut separator = \"\";\nfor &(flag, name) in Self::NAMES {{\nif self.contains(flag) {{\nf.write_str(separator)?;\nf.write_str(name)?;\nseparator = {:?};\n}}\n}}\nOk(())\n}}\n}}", flags.separator).unwrap();
}
