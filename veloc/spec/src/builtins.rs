//! Checked trait and memory-effect vocabulary. Type definitions live in types.rs.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::model::{Fields, list, name};
use crate::syntax::{Kind, Node, Record};

#[derive(Debug)]
pub(crate) struct Flag {
    pub name: String,
    pub bit: u8,
}

#[derive(Debug)]
pub(crate) struct Effect {
    pub reads: u8,
    pub writes: u8,
}

impl Effect {
    pub fn is_none(&self) -> bool {
        self.reads == 0 && self.writes == 0
    }
}

#[derive(Debug, Default)]
pub(crate) struct Builtins {
    pub traits: Vec<Flag>,
    pub regions: Vec<Flag>,
    pub effects: BTreeMap<String, Effect>,
}

impl Builtins {
    pub fn compile(records: &[Record], source: &str) -> Result<Self, Error> {
        let mut defs = Self::default();
        let mut effects = Vec::new();
        for record in records.iter().filter(|r| Self::is_definition(&r.kind)) {
            let mut fields = Fields::new(source, record.clone());
            match record.kind.as_str() {
                "trait" | "region" => {
                    let flags = if record.kind == "trait" {
                        &mut defs.traits
                    } else {
                        &mut defs.regions
                    };
                    let limit = if record.kind == "trait" { 16 } else { 8 };
                    let bit = number(source, fields.take("bit")?)?;
                    if record.name != record.name.to_ascii_uppercase()
                        || matches!(record.name.as_str(), "NONE" | "ALL" | "NAMES")
                    {
                        return Err(fields.error(
                            "flag names must be uppercase and cannot be NONE, ALL or NAMES",
                        ));
                    }
                    if bit >= limit || flags.iter().any(|f| u32::from(f.bit) == bit) {
                        return Err(
                            fields.error(format!("flag bit must be unique and less than {limit}"))
                        );
                    }
                    flags.push(Flag {
                        name: record.name.clone(),
                        bit: bit as u8,
                    });
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
        matches!(kind, "trait" | "region" | "effect")
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
            .map(|n| {
                self.reference(source, n, "trait", |n| {
                    self.traits.iter().any(|t| t.name == n)
                })
            })
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

    pub fn all_regions(&self) -> u8 {
        self.regions.iter().fold(0, |bits, r| bits | (1 << r.bit))
    }

    fn region_set(&self, source: &str, node: Node) -> Result<u8, Error> {
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
                    .regions
                    .iter()
                    .find(|r| r.name == name)
                    .ok_or_else(|| {
                        Error::at(
                            source,
                            node.offset,
                            format!("unknown memory region `{name}`"),
                        )
                    })?;
                1 << region.bit
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
