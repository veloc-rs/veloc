//! Checked physical representation of MIR types, separate from type semantics.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::Error;
use crate::model::{Fields, list, name};
use crate::syntax::{Kind, Record};

#[derive(Debug, Clone, Copy)]
struct Field {
    bits: u32,
    shift: u32,
}

impl Field {
    fn max(self) -> u16 {
        ((1u32 << self.bits) - 1) as u16
    }

    fn mask(self) -> u16 {
        self.max() << self.shift
    }
}

pub(crate) struct TypeEncoding {
    fields: BTreeMap<String, Field>,
    used: u16,
    pub codes: BTreeMap<String, ScalarCode>,
}

pub(crate) struct ScalarCode {
    pub offset: usize,
    pub code: u8,
}

impl TypeEncoding {
    pub fn compile(records: &[Record], source: &str) -> Result<Self, Error> {
        let mut encoding = None;
        for record in records.iter().filter(|r| r.kind == "encoding") {
            if record.name != "Type" {
                return Err(Error::at(
                    source,
                    record.offset,
                    "the MIR adapter only supports encoding Type",
                ));
            }
            if encoding.replace(record).is_some() {
                return Err(Error::at(source, record.offset, "duplicate encoding Type"));
            }
        }
        let record =
            encoding.ok_or_else(|| Error::at(source, 0, "missing encoding Type definition"))?;
        let mut fields = Fields::new(source, record.clone());
        let storage = fields.take("storage")?;
        if name(source, storage.clone())? != "u16" {
            return Err(Error::at(
                source,
                storage.offset,
                "MIR raw Type APIs require u16 storage",
            ));
        }
        let mut layout = BTreeMap::new();
        let mut shift = 0;
        let mut used = 0;
        for node in list(source, fields.take("fields")?)? {
            let fail = |message| Error::at(source, node.offset, message);
            let Kind::Call(name, args) = node.kind else {
                return Err(fail("expected encoding field(bits)"));
            };
            let [arg] = args.as_slice() else {
                return Err(fail("expected encoding field(bits)"));
            };
            let Kind::Number(bits) = arg.kind else {
                return Err(fail("encoding field width must be a number"));
            };
            if bits == 0 || bits > u16::BITS || shift + bits > u16::BITS {
                return Err(fail(
                    "encoding fields must have positive widths and fit u16",
                ));
            }
            // These limits belong to the Rust adapter, not the selected layout:
            // Scalar codes fit u8, lane_count returns u16, scalable is bool.
            let valid = match name.as_str() {
                "scalar" => bits <= u8::BITS,
                "lanes_log2" => (1u32 << bits) - 1 < u16::BITS,
                "scalable" => bits == 1,
                _ => return Err(fail("unknown Type encoding field")),
            };
            if !valid {
                return Err(fail(
                    "encoding field width exceeds its MIR API representation",
                ));
            }
            let field = Field { bits, shift };
            if layout.insert(name, field).is_some() {
                return Err(fail("duplicate Type encoding field"));
            }
            used |= field.mask();
            shift += bits;
        }
        for required in ["scalar", "lanes_log2", "scalable"] {
            if !layout.contains_key(required) {
                return Err(fields.error(format!("Type encoding is missing field `{required}`")));
            }
        }
        let mut codes = BTreeMap::new();
        let mut used_codes = BTreeSet::new();
        for node in list(source, fields.take("codes")?)? {
            let fail = |message| Error::at(source, node.offset, message);
            let Kind::Call(name, args) = node.kind else {
                return Err(fail("expected scalar encoding TypeName(code)".into()));
            };
            let [arg] = args.as_slice() else {
                return Err(fail("expected scalar encoding TypeName(code)".into()));
            };
            let Kind::Number(code) = arg.kind else {
                return Err(fail("scalar encoding code must be a number".into()));
            };
            let max = layout["scalar"].max();
            if code == 0 || code > u32::from(max) || !used_codes.insert(code) {
                return Err(fail(format!(
                    "scalar code must be unique and in 1..={max} (Type encoding)"
                )));
            }
            if codes
                .insert(
                    name.clone(),
                    ScalarCode {
                        offset: node.offset,
                        code: code as u8,
                    },
                )
                .is_some()
            {
                return Err(fail(format!("duplicate scalar encoding for `{name}`")));
            }
        }
        fields.finish()?;
        Ok(Self {
            fields: layout,
            used,
            codes,
        })
    }

    pub fn lanes_log2_max(&self) -> u32 {
        u32::from(self.fields["lanes_log2"].max())
    }

    pub fn generate(&self) -> String {
        let mut out = String::from(
            "// @generated by veloc-opgen; edit defs/types.ops.\n/// Compact MIR type representation. Unused high bits are reserved.\n",
        );
        let mut fields = self.fields.iter().collect::<Vec<_>>();
        fields.sort_by_key(|(_, field)| field.shift);
        for (name, field) in fields {
            writeln!(
                out,
                "/// Bits {}..{}: `{name}`.",
                field.shift,
                field.shift + field.bits
            )
            .unwrap();
        }
        out.push_str(
            "#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]\npub struct Type(u16);\n",
        );
        for (name, field) in &self.fields {
            let name = name.to_ascii_uppercase();
            writeln!(out, "const {name}_MASK: u16 = {:#06x};", field.mask()).unwrap();
            // Boolean flags use their mask directly; no unused shift/limit tables.
            if name != "SCALABLE" {
                writeln!(out, "const {name}_SHIFT: u32 = {};", field.shift).unwrap();
            }
        }
        writeln!(
            out,
            "const LANES_LOG2_MAX: u16 = {};",
            self.fields["lanes_log2"].max()
        )
        .unwrap();
        writeln!(out, "const USED_MASK: u16 = {:#06x};", self.used).unwrap();
        out.push_str("impl Type {\n");
        for name in self.fields.keys() {
            let field = name.to_ascii_uppercase();
            let (visibility, method, result) = match name.as_str() {
                "scalar" => ("pub(crate) ", "element_code", "u8"),
                "lanes_log2" => ("", "lanes_log2", "u16"),
                "scalable" => ("pub ", "is_scalable", "bool"),
                _ => unreachable!("Type encoding fields have been checked"),
            };
            let value = match result {
                "bool" => format!("self.0 & {field}_MASK != 0"),
                "u8" => format!("((self.0 & {field}_MASK) >> {field}_SHIFT) as u8"),
                _ => format!("(self.0 & {field}_MASK) >> {field}_SHIFT"),
            };
            writeln!(
                out,
                "{visibility}const fn {method}(self) -> {result} {{ {value} }}"
            )
            .unwrap();
        }
        out.push_str("/// Raw compact encoding; not stable across layout changes.\npub const fn to_raw(self) -> u16 { self.0 }\n");
        out.push_str("/// Return the element type, or self for scalars and INVALID.\npub const fn element_type(self) -> Self { Self(self.0 & SCALAR_MASK) }\n}\n");
        out
    }
}
