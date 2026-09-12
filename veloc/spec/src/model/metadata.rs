//! Typed metadata. Foreign values remain expressions evaluated by Rust.
use crate::Error;
use crate::model::data::Types;
use crate::model::expr::{Expr, Library};
use crate::model::records::{PropertyType, RecordField};
use crate::syntax::{Kind, Node, Record};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(crate) struct Metadata {
    pub name: String,
    pub fields: BTreeMap<String, Expr>,
    pub checks: Vec<(Expr, String)>,
    pub value_only: Option<Expr>,
}

impl Metadata {
    pub fn rust(&self, prefix: &str) -> String {
        let fields = self
            .fields
            .iter()
            .map(|(name, expr)| format!("{name}: {}", expr.const_rust(prefix)))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{prefix}{} {{ {fields} }}", self.name)
    }
}

pub(crate) struct Pending {
    node: Node,
    traits: Option<RecordField>,
    memory: Option<RecordField>,
}

fn name(offset: usize, value: &str) -> Node {
    Node {
        offset,
        kind: Kind::Name(value.into()),
    }
}
fn member(offset: usize, owner: &str, value: &str) -> Node {
    name(offset, &format!("{owner}::{value}"))
}
fn method(receiver: Node, method: &str, args: Vec<Node>) -> Node {
    Node {
        offset: receiver.offset,
        kind: Kind::Method(Box::new(receiver), method.into(), args),
    }
}

impl Pending {
    pub fn new(source: &str, node: Node, types: &Types) -> Result<Self, Error> {
        let Kind::Object(name, fields) = &node.kind else {
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
        if types.contains_value(name) {
            return Err(Error::at(
                source,
                node.offset,
                "metadata cannot contain SSA Value fields",
            ));
        }
        for (key, value) in fields {
            if !record.fields.iter().any(|f| f.name == *key) {
                return Err(Error::at(
                    source,
                    value.offset,
                    format!("unknown field `{key}` in {name}"),
                ));
            }
        }
        let contract = types.rust.analysis.as_ref();
        let field = |ty: Option<&String>| -> Result<Option<RecordField>, Error> {
            let mut fields = record.fields.iter().filter(|f| {
                matches!(&f.ty, PropertyType::Named(n) if ty.is_some_and(|t| types.rust.rust(n) == types.rust.rust(t)))
            });
            let first = fields.next().cloned();
            if fields.next().is_some() {
                return Err(Error::at(
                    source,
                    node.offset,
                    "ambiguous operation analysis fields",
                ));
            }
            Ok(first)
        };
        Ok(Self {
            traits: field(contract.map(|m| &m.traits))?,
            memory: field(contract.map(|m| &m.memory))?,
            node,
        })
    }

    pub fn finish(
        self,
        source: &str,
        types: &Types,
        expressions: &mut Library,
        env: &BTreeMap<String, Expr>,
        vocabulary: super::Vocabulary<'_>,
        traits: &BTreeSet<String>,
        semantic: bool,
    ) -> Result<Metadata, Error> {
        let offset = self.node.offset;
        if self.traits.is_none() && !traits.is_empty() {
            return Err(Error::at(
                source,
                offset,
                "derived traits require an OpTraits metadata field",
            ));
        }
        if self.memory.is_none() && semantic {
            return Err(Error::at(
                source,
                offset,
                "semantics require a MemoryEffect metadata field",
            ));
        }
        let Kind::Object(name_, mut fields) = self.node.kind else {
            unreachable!()
        };
        let record = types
            .records
            .iter()
            .find(|r| r.name == name_)
            .expect("checked metadata record");
        let contract = types.rust.analysis.as_ref();
        let mut checks = Vec::new();
        if let Some(field) = &self.traits {
            let contract = contract.expect("analysis binding");
            let declared = fields.remove(&field.name).unwrap_or_else(|| Node {
                offset,
                kind: Kind::Call(format!("{}::empty", contract.traits), vec![]),
            });
            let mut value = declared;
            for flag in traits {
                value = method(value, "union", vec![member(offset, &contract.traits, flag)]);
            }
            let contains = |flag: &str| {
                method(
                    value.clone(),
                    "contains",
                    vec![member(offset, &contract.traits, flag)],
                )
            };
            let not = |v| Node {
                offset,
                kind: Kind::Unary("!", Box::new(v)),
            };
            checks.push((
                Node {
                    offset,
                    kind: Kind::Binary(
                        "||",
                        Box::new(not(contains("ABORT"))),
                        Box::new(contains("TERMINATOR")),
                    ),
                },
                "ABORT requires TERMINATOR".into(),
            ));
            if semantic {
                // Derived facts are trusted primitive facts. Opaque user expressions
                // must not add unproved algebraic/control/trap promises.
                for flag in [
                    "COMMUTATIVE",
                    "ASSOCIATIVE",
                    "IDEMPOTENT",
                    "TERMINATOR",
                    "MAY_TRAP",
                ] {
                    if !traits.contains(flag) {
                        checks.push((
                            not(contains(flag)),
                            format!("semantics do not justify {flag}"),
                        ));
                    }
                }
            }
            fields.insert(field.name.clone(), value);
        }
        if let Some(field) = &self.memory {
            let contract = contract.expect("analysis binding");
            if !fields.contains_key(&field.name) {
                if !semantic {
                    return Err(Error::at(
                        source,
                        offset,
                        "unmodeled operations must declare their memory effect",
                    ));
                }
                fields.insert(field.name.clone(), contract.pure.clone());
            }
            if semantic {
                checks.push((
                    method(fields[&field.name].clone(), "is_none", vec![]),
                    "executable semantics require no memory effects".into(),
                ));
            }
        }
        let value_only = if let (Some(traits), Some(memory), Some(contract)) =
            (&self.traits, &self.memory, contract)
        {
            let control = member(offset, &contract.traits, "TERMINATOR");
            let flags = method(fields[&traits.name].clone(), "contains", vec![control]);
            let memory = method(fields[&memory.name].clone(), "is_none", vec![]);
            let node = Node {
                offset,
                kind: Kind::Binary(
                    "&&",
                    Box::new(memory),
                    Box::new(Node {
                        offset,
                        kind: Kind::Unary("!", Box::new(flags)),
                    }),
                ),
            };
            Some(expressions.metadata_field(
                source,
                &node,
                &PropertyType::Named("bool".into()),
                env,
                vocabulary,
            )?)
        } else {
            None
        };
        let values = record
            .fields
            .iter()
            .map(|field| {
                let node = fields.remove(&field.name).ok_or_else(|| {
                    Error::at(
                        source,
                        offset,
                        format!("missing field `{}` in {name_}", field.name),
                    )
                })?;
                let expr = expressions.metadata_field(source, &node, &field.ty, env, vocabulary)?;
                Ok((field.name.clone(), expr))
            })
            .collect::<Result<_, Error>>()?;
        let checks = checks
            .into_iter()
            .map(|(node, message)| {
                Ok((
                    expressions.metadata_field(
                        source,
                        &node,
                        &PropertyType::Named("bool".into()),
                        env,
                        vocabulary,
                    )?,
                    message,
                ))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Metadata {
            name: name_,
            fields: values,
            checks,
            value_only,
        })
    }
}

/// A compilation unit has one metadata struct type; each opcode has a constant value.
pub(crate) fn record_type(ops: &[crate::model::Op]) -> Option<&str> {
    ops.first().map(|op| op.meta.name.as_str())
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

pub(crate) fn value_contract(ops: &[crate::model::Op], opcode: &str, prefix: &str) -> String {
    use std::fmt::Write;
    let mut out =
        format!("impl {opcode} {{ pub const fn is_value_only(self) -> bool {{ match self {{");
    for op in ops {
        let value = op
            .meta
            .value_only
            .as_ref()
            .map_or("false".into(), |e| e.const_rust(prefix));
        writeln!(
            out,
            "Self::{} => {{ const VALUE: bool = {value}; VALUE }},",
            op.name
        )
        .unwrap();
    }
    out.push_str("} } }\n");
    out
}

#[derive(Debug, Clone)]
pub(crate) struct Analysis {
    pub traits: String,
    pub memory: String,
    pub pure: Node,
}

impl Analysis {
    pub fn compile(records: &[Record], source: &str) -> Result<Option<Self>, Error> {
        let mut traits = None;
        let mut memory = None;
        for record in records
            .iter()
            .filter(|r| crate::interfaces::rust_binding(r).is_some())
        {
            if let Some(node) = record.fields.get("analysis") {
                match &node.kind {
                    Kind::Name(n) if n == "traits" && traits.is_none() => {
                        traits = Some(record.name.clone())
                    }
                    Kind::Call(n, args) if n == "memory" && args.len() == 1 && memory.is_none() => {
                        memory = Some((record.name.clone(), args[0].clone()))
                    }
                    _ => {
                        return Err(Error::at(
                            source,
                            node.offset,
                            "expected one traits or memory(pure) analysis contract",
                        ));
                    }
                }
            }
        }
        let metadata = match (traits, memory) {
            (None, None) => None,
            (Some(traits), Some((memory, pure))) => Some(Analysis {
                traits,
                memory,
                pure,
            }),
            _ => {
                return Err(Error::at(
                    source,
                    0,
                    "analysis requires both traits and memory contracts",
                ));
            }
        };
        Ok(metadata)
    }
}
