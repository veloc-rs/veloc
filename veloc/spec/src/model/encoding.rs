//! Checked bit layouts for explicitly declared data fields.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::Error;
use crate::model::{Fields, identifier, list, name};
use crate::syntax::{Kind, Node};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Storage {
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl Storage {
    pub fn parse(source: &str, node: Node) -> Result<Self, Error> {
        let offset = node.offset;
        match name(source, node)?.as_str() {
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "u128" => Ok(Self::U128),
            _ => Err(Error::at(
                source,
                offset,
                "bit storage must be u8, u16, u32, u64 or u128",
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
        }
    }

    fn bits(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::U32 => 32,
            Self::U64 => 64,
            Self::U128 => 128,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Field {
    bits: u32,
    shift: u32,
}

impl Field {
    fn max(self) -> u128 {
        u128::MAX >> (128 - self.bits)
    }

    fn mask(self) -> u128 {
        self.max() << self.shift
    }
}

/// Shared layout checks for explicit bit fields and flags.
#[derive(Debug)]
pub(crate) struct BitLayout {
    storage: Storage,
    fields: BTreeMap<String, Field>,
    pub used: u128,
}

impl BitLayout {
    pub fn packed(source: &str, fields: &mut Fields<'_>) -> Result<Self, Error> {
        let storage = Storage::parse(source, fields.take("storage")?)?;
        Self::packed_fields(source, storage, fields.take("fields")?)
    }

    fn packed_fields(source: &str, storage: Storage, node: Node) -> Result<Self, Error> {
        let mut layout = Self::new(storage);
        let mut shift = 0;
        for node in list(source, node)? {
            let fail = |message| Error::at(source, node.offset, message);
            let Kind::Call(name, args) = node.kind else {
                return Err(fail("expected encoding field(bits)"));
            };
            identifier(source, node.offset, &name)?;
            let [arg] = args.as_slice() else {
                return Err(fail("expected encoding field(bits)"));
            };
            let Kind::Number(bits) = arg.kind else {
                return Err(fail("encoding field width must be a number"));
            };
            layout.insert(source, node.offset, &name, bits, shift)?;
            shift += bits;
        }
        Ok(layout)
    }

    pub fn check_methods(&self, source: &str, offset: usize) -> Result<(), Error> {
        let mut names = BTreeSet::from(["empty".to_owned()]);
        for (name, field) in &self.fields {
            let getter = if field.bits == 1 {
                format!("is_{name}")
            } else {
                name.clone()
            };
            for method in [getter, format!("with_{name}")] {
                identifier(source, offset, &method)?;
                if !names.insert(method.clone()) {
                    return Err(Error::at(
                        source,
                        offset,
                        format!("duplicate encoding method `{method}`"),
                    ));
                }
            }
            if field.bits > 1 && !names.insert(format!("{}_MAX", name.to_ascii_uppercase())) {
                return Err(Error::at(source, offset, "duplicate encoding limit"));
            }
        }
        Ok(())
    }

    /// Plain packed records expose fields, not flag-set operations.
    pub fn generate(&self, ty: &str) -> String {
        let mut out = String::new();
        let repr = self.storage.name();
        writeln!(out, "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]\npub struct {ty}({repr});\nimpl {ty} {{\npub const fn empty() -> Self {{ Self(0) }}").unwrap();
        for (name, field) in &self.fields {
            let mask = field.mask();
            let shift = field.shift;
            let full = field.bits == self.storage.bits();
            let value = if field.bits == 1 {
                format!("value as {repr}")
            } else {
                "value".into()
            };
            let value = if shift == 0 {
                value
            } else {
                format!("(({value}) << {shift})")
            };
            let update = if full {
                value
            } else {
                format!("(self.0 & !{mask}) | ({value})")
            };
            if field.bits == 1 {
                writeln!(out, "pub const fn is_{name}(self) -> bool {{ self.0 & {mask} != 0 }}\npub const fn with_{name}(self, value: bool) -> Self {{ Self({update}) }}").unwrap();
            } else {
                let max = field.max();
                let constant = name.to_ascii_uppercase();
                let read = if full {
                    "self.0".into()
                } else if shift == 0 {
                    format!("self.0 & {mask}")
                } else {
                    format!("(self.0 & {mask}) >> {shift}")
                };
                let check = if full {
                    String::new()
                } else {
                    format!(
                        "assert!(value <= Self::{constant}_MAX, \"encoding field out of range\");"
                    )
                };
                writeln!(out, "pub const {constant}_MAX: {repr} = {max};\npub const fn {name}(self) -> {repr} {{ {read} }}\npub const fn with_{name}(self, value: {repr}) -> Self {{ {check} Self({update}) }}").unwrap();
            }
        }
        out.push_str("}\n");
        out
    }

    pub fn new(storage: Storage) -> Self {
        Self {
            storage,
            fields: BTreeMap::new(),
            used: 0,
        }
    }

    pub fn insert(
        &mut self,
        source: &str,
        offset: usize,
        name: &str,
        bits: u32,
        shift: u32,
    ) -> Result<(), Error> {
        if bits == 0 || bits > self.storage.bits() || shift > self.storage.bits() - bits {
            return Err(Error::at(
                source,
                offset,
                format!(
                    "bit fields must have positive widths and fit {}",
                    self.storage.name()
                ),
            ));
        }
        if self.fields.contains_key(name) {
            return Err(Error::at(
                source,
                offset,
                format!("duplicate bit field `{name}`"),
            ));
        }
        let field = Field { bits, shift };
        if self.used & field.mask() != 0 {
            return Err(Error::at(
                source,
                offset,
                format!("bit field `{name}` overlaps another field"),
            ));
        }
        self.used |= field.mask();
        self.fields.insert(name.to_owned(), field);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_layout_handles_full_width_and_rejects_invalid_ranges() {
        for storage in [
            Storage::U8,
            Storage::U16,
            Storage::U32,
            Storage::U64,
            Storage::U128,
        ] {
            let width = storage.bits();
            let mut layout = BitLayout::new(storage);
            // Splitting at the highest bit exercises both multi-bit encodings
            // and the one-bit fields used by flag sets.
            layout.insert("", 0, "low", width - 1, 0).unwrap();
            layout.insert("", 0, "high", 1, width - 1).unwrap();
            let all = u128::MAX >> (128 - width);
            assert_eq!(layout.used, all);
            assert!(layout.insert("", 0, "overlap", 1, 0).is_err());
            assert_eq!(layout.used, all);

            let mut full = BitLayout::new(storage);
            full.insert("", 0, "full", width, 0).unwrap();
            assert_eq!(full.used, all);
            assert_eq!(full.fields["full"].max(), all);
            for (bits, shift) in [
                (0, 0),
                (width + 1, 0),
                (1, width),
                (width, 1),
                (u32::MAX, u32::MAX),
            ] {
                let mut invalid = BitLayout::new(storage);
                assert!(invalid.insert("", 0, "bad", bits, shift).is_err());
                assert_eq!(invalid.used, 0);
                assert!(invalid.fields.is_empty());
            }
        }
    }
}
