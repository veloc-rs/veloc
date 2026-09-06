//! Named flag sets and the MIR trait/memory-effect vocabulary.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::encoding::{BitLayout, Storage};
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
pub(crate) struct Effect {
    pub reads: u128,
    pub writes: u128,
}

impl Effect {
    pub fn is_none(&self) -> bool {
        self.reads == 0 && self.writes == 0
    }
}

#[derive(Debug, Default)]
pub(crate) struct Builtins {
    pub encodings: BTreeMap<String, BitLayout>,
    pub flags: BTreeMap<String, Flags>,
    pub effects: BTreeMap<String, Effect>,
}

impl Builtins {
    pub fn compile(records: &[Record], source: &str) -> Result<Self, Error> {
        let mut defs = Self::default();
        let mut effects = Vec::new();
        for record in records
            .iter()
            .filter(|r| Self::is_definition(&r.kind) && !(r.kind == "encoding" && r.name == "Type"))
        {
            let mut fields = Fields::new(source, record.clone());
            match record.kind.as_str() {
                "flags" | "encoding" => {
                    if matches!(
                        record.name.as_str(),
                        "Opcode"
                            | "OpFormat"
                            | "TypeClass"
                            | "MemoryEffect"
                            | "OpSpec"
                            | "TypeError"
                            | "type_rules"
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
                "effect" => {
                    if record.name != record.name.to_ascii_uppercase() {
                        return Err(fields.error("effect names must be uppercase"));
                    }
                    effects.push((
                        record.offset,
                        record.name.clone(),
                        fields.take("reads")?,
                        fields.take("writes")?,
                    ));
                }
                _ => unreachable!(),
            }
            fields.finish()?;
        }
        for (offset, name, reads, writes) in effects {
            let effect = Effect {
                reads: defs.region_set(source, reads)?,
                writes: defs.region_set(source, writes)?,
            };
            if name == "NONE" && !effect.is_none() {
                return Err(Error::at(
                    source,
                    offset,
                    "NONE must have no memory effects",
                ));
            }
            if name == "UNKNOWN"
                && (effect.reads != defs.all_regions() || effect.writes != defs.all_regions())
            {
                return Err(Error::at(
                    source,
                    offset,
                    "UNKNOWN must read and write all regions",
                ));
            }
            defs.effects.insert(name, effect);
        }
        Ok(defs)
    }

    pub fn is_definition(kind: &str) -> bool {
        matches!(kind, "flags" | "effect" | "encoding")
    }

    pub fn effect(&self, source: &str, node: Node) -> Result<String, Error> {
        self.reference(source, node, "memory effect", |n| {
            self.effects.contains_key(n)
        })
    }

    pub fn traits(&self, source: &str, node: Node) -> Result<Vec<String>, Error> {
        let offset = node.offset;
        let values = list(source, node)?
            .into_iter()
            .map(|n| self.reference(source, n, "trait", |n| self.has_trait(n)))
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
            return Err(Error::at(source, offset, "duplicate trait"));
        }
        Ok(values)
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
        self.flags
            .get("OpTraits")
            .is_some_and(|flags| flags.members.iter().any(|flag| flag.name == name))
    }

    fn all_regions(&self) -> u128 {
        self.flags.get("MemoryRegions").map_or(0, Flags::all)
    }

    fn region_set(&self, source: &str, node: Node) -> Result<u128, Error> {
        let mut bits = 0;
        let mut seen = BTreeSet::new();
        let values = list(source, node)?;
        for node in &values {
            let name = name(source, node.clone())?;
            if !seen.insert(name.clone()) {
                return Err(Error::at(source, node.offset, "duplicate memory region"));
            }
            let value = if name == "ALL" {
                if values.len() != 1 {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "ALL must be the only region",
                    ));
                }
                self.all_regions()
            } else {
                let region = self
                    .flags
                    .get("MemoryRegions")
                    .and_then(|flags| flags.members.iter().find(|r| r.name == name))
                    .ok_or_else(|| {
                        Error::at(
                            source,
                            node.offset,
                            format!("unknown memory region `{name}`"),
                        )
                    })?;
                1u128 << region.bit
            };
            bits |= value;
        }
        Ok(bits)
    }
}

fn number(source: &str, node: Node) -> Result<u32, Error> {
    match node.kind {
        Kind::Number(value) => Ok(value),
        _ => Err(Error::at(source, node.offset, "expected a number")),
    }
}
