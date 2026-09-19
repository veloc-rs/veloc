//! Typed, cross-dialect value rules. Syntax and operation contracts come from
//! OpSpec; this module knows neither MIR storage nor machine registers.
mod check;
mod decision;
pub(crate) mod equivalence;
mod functions;
mod rust;
pub(crate) mod typed;
pub use decision::{DecisionRust, decisions};

use crate::schema::Operation;
use crate::syntax::{Kind, Node};
use crate::{Definitions, Error};
pub use rust::Rust;
use std::collections::{BTreeMap, BTreeSet};

/// Logical namespaces are supplied by the embedding build, not globally known.
#[derive(Default)]
pub struct Dialects {
    operations: BTreeMap<String, Operation>,
}

impl Dialects {
    pub fn insert(&mut self, namespace: &str, defs: &Definitions) -> Result<(), Error> {
        if !identifier(namespace) {
            return Err(Error::at("", 0, "invalid dialect identifier"));
        }
        if self
            .operations
            .keys()
            .any(|key| key.starts_with(&format!("{namespace}.")))
        {
            return Err(Error::at("", 0, format!("duplicate dialect {namespace}")));
        }
        for op in defs.operations() {
            self.operations
                .insert(format!("{namespace}.{}", op.name), op);
        }
        Ok(())
    }

    fn operation(&self, name: &str, source: &str, offset: usize) -> Result<&Operation, Error> {
        self.operations
            .get(name)
            .ok_or_else(|| Error::at(source, offset, format!("unknown operation {name}")))
    }
}

pub(crate) fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[derive(Debug, Clone)]
struct Call {
    op: String,
    args: Vec<Expr>,
    offset: usize,
}

#[derive(Debug, Clone)]
enum Expr {
    Value(String, usize),
    Call(Call),
}

fn expression(source: &str, node: Node) -> Result<Expr, Error> {
    match node.kind {
        Kind::Name(name) => Ok(Expr::Value(name, node.offset)),
        Kind::Method(receiver, op, args) => {
            let Kind::Name(namespace) = receiver.kind else {
                return Err(Error::at(
                    source,
                    node.offset,
                    "expected dialect.operation(...)",
                ));
            };
            Ok(Expr::Call(Call {
                op: format!("{namespace}.{op}"),
                args: args
                    .into_iter()
                    .map(|arg| expression(source, arg))
                    .collect::<Result<_, _>>()?,
                offset: node.offset,
            }))
        }
        _ => Err(Error::at(
            source,
            node.offset,
            "expected a bound value or dialect.operation(...)",
        )),
    }
}

#[derive(Debug)]
struct Rule {
    name: String,
    root: Call,
    outputs: Vec<Expr>,
}

fn parse(source: &str, declarations: &[crate::syntax::Decl]) -> Result<Vec<Rule>, Error> {
    let mut rules = Vec::new();
    let mut names = BTreeSet::new();
    for mut record in declarations.iter().cloned() {
        if !matches!(&record.kind, crate::syntax::DeclKind::Fields(kind) if kind == "rule") {
            return Err(Error::at(
                source,
                record.offset,
                "expected rule declaration",
            ));
        }
        if !names.insert(record.name.clone()) {
            return Err(Error::at(source, record.offset, "duplicate rule name"));
        }
        for (field, node) in &record.fields {
            if !matches!(field.as_str(), "match" | "emit") {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown rule field {field}"),
                ));
            }
        }
        let root = record
            .fields
            .remove("match")
            .ok_or_else(|| Error::at(source, record.offset, "missing match"))?;
        let Expr::Call(root) = expression(source, root)? else {
            return Err(Error::at(
                source,
                record.offset,
                "match must name an operation",
            ));
        };
        let emit = record
            .fields
            .remove("emit")
            .ok_or_else(|| Error::at(source, record.offset, "missing emit"))?;
        let nodes = match emit.kind {
            Kind::List(nodes) => nodes,
            _ => vec![emit],
        };
        rules.push(Rule {
            name: record.name,
            root,
            outputs: nodes
                .into_iter()
                .map(|node| expression(source, node))
                .collect::<Result<_, _>>()?,
        });
    }
    Ok(rules)
}

/// A checked plan, not executable bytecode. Backends generate ordinary Rust
/// from it; no rule interpreter or semantic evaluator runs during lowering.
pub struct Program {
    rules: BTreeMap<String, check::CheckedRule>,
}

impl Program {
    pub fn compile(source: &str, dialects: &Dialects) -> Result<Self, Error> {
        Self::from_declarations(source, &crate::syntax::parse(source)?, dialects)
    }

    pub(crate) fn from_declarations(
        source: &str,
        declarations: &[crate::syntax::Decl],
        dialects: &Dialects,
    ) -> Result<Self, Error> {
        let mut rules = BTreeMap::new();
        for rule in parse(source, declarations)? {
            let root = rule.root.op.clone();
            if rules.contains_key(&root) {
                return Err(Error::at(
                    source,
                    rule.root.offset,
                    format!("overlapping unconditional rules for {root}"),
                ));
            }
            rules.insert(root, check::check(source, dialects, rule)?);
        }
        Ok(Self { rules })
    }

    /// Discover exact primitive matches, then pass them through the same type
    /// checker as explicit rules. Explicit rules take precedence. This is not
    /// an equivalence proof for general expressions or target legalization.
    pub fn infer_primitives(
        &mut self,
        dialects: &Dialects,
        from: &str,
        to: &str,
    ) -> Result<(), Error> {
        for (name, op) in &dialects.operations {
            if !name.starts_with(&format!("{from}.")) || self.rules.contains_key(name) {
                continue;
            }
            let (Some(primitive), Ok(signature)) = (&op.primitive, &op.signature) else {
                continue;
            };
            let mut candidates = Vec::new();
            for (target, candidate) in &dialects.operations {
                if !target.starts_with(&format!("{to}."))
                    || candidate.primitive.as_ref() != Some(primitive)
                {
                    continue;
                }
                let args: Vec<_> = (0..signature.inputs.len())
                    .map(|i| Expr::Value(format!("arg{i}"), 0))
                    .collect();
                let rule = Rule {
                    name: format!("inferred_{name}"),
                    root: Call {
                        op: name.clone(),
                        args: args.clone(),
                        offset: 0,
                    },
                    outputs: vec![Expr::Call(Call {
                        op: target.clone(),
                        args,
                        offset: 0,
                    })],
                };
                if let Ok(rule) = check::check("", dialects, rule) {
                    candidates.push(rule);
                }
            }
            if candidates.len() > 1 {
                return Err(Error::at(
                    "",
                    0,
                    format!("ambiguous primitive lowering for {name}; write an explicit rule"),
                ));
            }
            if let Some(rule) = candidates.pop() {
                self.rules.insert(name.clone(), rule);
            }
        }
        Ok(())
    }

    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.rules
            .values()
            .flat_map(|rule| rule.insts.iter().map(|inst| inst.op.as_str()))
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    pub fn roots(&self) -> impl Iterator<Item = &str> {
        self.rules.keys().map(String::as_str)
    }
}
