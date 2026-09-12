//! Domain projections over generic declaration values.
//!
//! Flags/enum/record parsing lives in `data`. This adapter knows what operation
//! traits and memory behaviors mean, including facts derived from semantics.
use crate::Error;
use crate::model::builtins::{Builtins, Effect};
use crate::model::data::{Types, Value};
use crate::model::records::PropertyType;
use crate::syntax::{Kind, Node};

pub(crate) struct Pending {
    node: Node,
    traits: Option<String>,
    memory: Option<String>,
}

impl Pending {
    pub fn new(source: &str, node: Node, types: &Types) -> Result<Self, Error> {
        let Kind::Object(name, _) = &node.kind else {
            return Err(Error::at(
                source,
                node.offset,
                "meta requires a typed record value",
            ));
        };
        let record = types
            .records
            .iter()
            .find(|r| r.name == *name)
            .ok_or_else(|| {
                Error::at(
                    source,
                    node.offset,
                    format!("unknown metadata struct `{name}`"),
                )
            })?;
        let field = |ty: &str| -> Result<Option<String>, Error> {
            let mut fields = record
                .fields
                .iter()
                .filter(|f| f.ty == PropertyType::Named(ty.into()));
            let first = fields.next().map(|f| f.name.clone());
            if fields.next().is_some() {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("operation metadata has ambiguous {ty} fields"),
                ));
            }
            Ok(first)
        };
        Ok(Self {
            traits: field("OpTraits")?,
            memory: field("MemoryEffect")?,
            node,
        })
    }

    pub fn explicit_memory(&self) -> bool {
        self.get(&self.memory).is_some()
    }

    fn get(&self, field: &Option<String>) -> Option<&Node> {
        let Kind::Object(_, fields) = &self.node.kind else {
            unreachable!()
        };
        fields.get(field.as_ref()?)
    }

    pub fn traits(
        &self,
        source: &str,
        types: &Types,
        builtins: &Builtins,
    ) -> Result<Vec<String>, Error> {
        let value = match self.get(&self.traits) {
            Some(node) => types.value(
                source,
                &PropertyType::Named("OpTraits".into()),
                node.clone(),
                builtins,
            )?,
            None => return Ok(Vec::new()),
        };
        let Value::Flags(_, members) = value else {
            return Err(Error::at(
                source,
                self.node.offset,
                "operation traits require an OpTraits flags declaration",
            ));
        };
        Ok(members)
    }

    pub fn memory(
        &self,
        source: &str,
        types: &Types,
        builtins: &Builtins,
    ) -> Result<Option<Effect>, Error> {
        let value = match self.get(&self.memory) {
            Some(node) => types.value(
                source,
                &PropertyType::Named("MemoryEffect".into()),
                node.clone(),
                builtins,
            )?,
            None => return Ok(None),
        };
        match value {
            Value::Variant(_, variant, args) if variant == "Unknown" && args.is_empty() => {
                Ok(Some(Effect::Unknown))
            }
            Value::Variant(_, variant, args) if variant == "Known" => match args.as_slice() {
                [Value::Flags(ty, members)] if ty == "MemoryEffects" => {
                    Ok(Some(Effect::Known(members.clone())))
                }
                _ => Err(Error::at(
                    source,
                    self.node.offset,
                    "Known must carry MemoryEffects",
                )),
            },
            _ => Err(Error::at(
                source,
                self.node.offset,
                "memory contract requires Known(MemoryEffects) or Unknown",
            )),
        }
    }

    pub fn finish(
        mut self,
        source: &str,
        types: &Types,
        builtins: &Builtins,
        traits: &[String],
        memory: &Effect,
    ) -> Result<Value, Error> {
        let offset = self.node.offset;
        let at = |kind| Node { offset, kind };
        let members = |names: &[String]| {
            at(Kind::List(
                names.iter().map(|n| at(Kind::Name(n.clone()))).collect(),
            ))
        };
        let Kind::Object(_, fields) = &mut self.node.kind else {
            unreachable!()
        };
        if let Some(name) = self.traits {
            fields.insert(name, members(traits));
        } else if !traits.is_empty() {
            return Err(Error::at(
                source,
                offset,
                "derived traits require an OpTraits metadata field",
            ));
        }
        if let Some(name) = self.memory {
            fields.insert(
                name,
                match memory {
                    Effect::Unknown => at(Kind::Name("Unknown".into())),
                    Effect::Known(names) => at(Kind::Call("Known".into(), vec![members(names)])),
                },
            );
        } else if !matches!(memory, Effect::Unknown) {
            return Err(Error::at(
                source,
                offset,
                "known memory behavior requires a MemoryEffect metadata field",
            ));
        }
        types.record(source, self.node, builtins)
    }

    pub fn has_memory_field(&self) -> bool {
        self.memory.is_some()
    }
}

/// A compilation unit has one metadata struct type; each opcode has a constant value.
pub(crate) fn record_type(ops: &[crate::model::Op]) -> Option<&str> {
    ops.first().map(|op| {
        let Value::Record(ty, _) = &op.meta else {
            unreachable!()
        };
        ty.as_str()
    })
}

pub(crate) fn generate(ops: &[crate::model::Op], opcode: &str, prefix: &str) -> String {
    use std::fmt::Write;
    let Some(ty) = record_type(ops) else {
        return String::new();
    };
    let mut out = format!(
        "impl {opcode} {{ pub const fn meta(self) -> &'static {prefix}{ty} {{ match self {{\n"
    );
    for op in ops {
        writeln!(
            out,
            "Self::{} => {{ const META: {prefix}{ty} = {}; &META }},",
            op.name,
            op.meta.rust(prefix)
        )
        .unwrap();
    }
    out.push_str("} } }\n");
    out
}
