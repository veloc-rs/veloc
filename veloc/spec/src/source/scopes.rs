//! Import visibility is lexical to each file, independently of loading order.
//! Resolution within a namespace remains the checked model's responsibility.
use std::collections::{BTreeMap, BTreeSet};

use crate::syntax::{FunctionBody, Kind, Node, Record, Results};
use crate::{Error, model::records::rust_binding};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Space {
    Data,
    Type,
    Function,
    Value,
}

pub(super) fn check(
    source: &str,
    records: &[Record],
    files: &[super::File],
    defs: &crate::Definitions,
) -> Result<(), Error> {
    let mut symbols = BTreeMap::<(Space, String), BTreeSet<usize>>::new();
    for (file, entry) in files.iter().enumerate() {
        for record in &records[entry.records.clone()] {
            let space = match record.kind.as_str() {
                "type" if rust_binding(record).is_some() => Space::Data,
                "type" | "typeset" => Space::Type,
                "fn" | "extern-fn" => Space::Function,
                "const" => Space::Value,
                "struct" | "enum" | "encoding" | "comparison" | "interface"
                | "extern-interface" => Space::Data,
                _ => continue,
            };
            symbols
                .entry((space, record.name.clone()))
                .or_default()
                .insert(file);
            if space == Space::Type && record.kind == "type" {
                symbols
                    .entry((Space::Type, format!("Type::{}", record.name)))
                    .or_default()
                    .insert(file);
            }
            if let Some(en) = defs.data.enums.iter().find(|en| en.name == record.name) {
                for (name, _) in &en.variants {
                    symbols
                        .entry((Space::Value, name.clone()))
                        .or_default()
                        .insert(file);
                }
            }
            if record.kind == "comparison" {
                for field in ["variants", "members", "predicates"] {
                    if let Some(Node {
                        kind: Kind::List(items),
                        ..
                    }) = record.fields.get(field)
                    {
                        for item in items {
                            if let Kind::Name(name) | Kind::Call(name, _) = &item.kind {
                                symbols
                                    .entry((Space::Value, name.clone()))
                                    .or_default()
                                    .insert(file);
                            }
                        }
                    }
                }
            }
        }
    }
    for file in files {
        let checker = Checker {
            source,
            defs,
            symbols: &symbols,
            visible: &file.visible,
        };
        for record in &records[file.records.clone()] {
            checker.record(record)?;
        }
    }
    Ok(())
}

struct Checker<'a> {
    defs: &'a crate::Definitions,
    source: &'a str,
    symbols: &'a BTreeMap<(Space, String), BTreeSet<usize>>,
    visible: &'a BTreeSet<usize>,
}

impl Checker<'_> {
    fn name(
        &self,
        name: &str,
        space: Space,
        offset: usize,
        locals: &BTreeSet<String>,
    ) -> Result<(), Error> {
        if !locals.contains(name)
            && let Some(owners) = self.symbols.get(&(space, name.to_owned()))
            && owners.is_disjoint(self.visible)
        {
            return Err(Error::at(
                self.source,
                offset,
                format!("`{name}` is not imported in this file"),
            ));
        }
        Ok(())
    }

    fn node(
        &self,
        node: &Node,
        space: Option<Space>,
        locals: &BTreeSet<String>,
    ) -> Result<(), Error> {
        match &node.kind {
            Kind::Name(name) => {
                if let Some((owner, _)) = name.split_once("::") {
                    self.name(owner, Space::Data, node.offset, locals)?;
                    self.name(name, Space::Value, node.offset, locals)?;
                    self.name(name, Space::Type, node.offset, locals)?;
                }
                if let Some(space) = space {
                    self.name(name, space, node.offset, locals)?;
                } else {
                    // Expression lookup prefers lexical values, then concrete IR types.
                    let space = if self.symbols.contains_key(&(Space::Type, name.clone())) {
                        Space::Type
                    } else if self.symbols.contains_key(&(Space::Value, name.clone())) {
                        Space::Value
                    } else {
                        Space::Data
                    };
                    self.name(name, space, node.offset, locals)?;
                }
            }
            Kind::Call(name, args) => {
                if let Some((owner, _)) = name.split_once("::") {
                    self.name(owner, Space::Data, node.offset, locals)?;
                }
                self.name(name, space.unwrap_or(Space::Function), node.offset, locals)?;
                if space.is_none() {
                    self.name(name, Space::Value, node.offset, locals)?;
                }
                for arg in args {
                    // Value(I32) embeds an IR type in a data-type declaration.
                    let inner = if name == "Value" {
                        Some(Space::Type)
                    } else {
                        space
                    };
                    self.node(arg, inner, locals)?;
                }
            }
            Kind::Method(receiver, name, args) => {
                if let Kind::Name(owner) = &receiver.kind
                    && !locals.contains(owner)
                {
                    self.name(owner, Space::Data, receiver.offset, locals)?;
                    self.name(
                        &format!("{owner}::{name}"),
                        Space::Function,
                        node.offset,
                        locals,
                    )?;
                }
                self.node(receiver, None, locals)?;
                for arg in args {
                    self.node(arg, None, locals)?;
                }
            }
            Kind::Member(receiver, name) => {
                self.node(receiver, None, locals)?;
                if let Kind::Name(owner) = &receiver.kind
                    && !locals.contains(owner)
                {
                    self.name(
                        &format!("{owner}::{name}"),
                        Space::Value,
                        node.offset,
                        locals,
                    )?;
                }
            }
            Kind::Object(name, fields) => {
                self.name(name, Space::Data, node.offset, locals)?;
                for value in fields.values() {
                    self.node(value, None, locals)?;
                }
            }
            Kind::List(nodes) | Kind::Union(nodes) | Kind::Intersection(nodes) => {
                for node in nodes {
                    self.node(node, space, locals)?;
                }
            }
            Kind::Unary(_, value) | Kind::Try(value) => self.node(value, None, locals)?,
            Kind::Binary(_, a, b) => {
                self.node(a, None, locals)?;
                self.node(b, None, locals)?;
            }
            Kind::Lambda(name, body) => {
                let mut locals = locals.clone();
                locals.insert(name.clone());
                self.node(body, None, &locals)?;
            }
            Kind::Text(_) | Kind::Number(_) | Kind::Integer(_) => {}
        }
        Ok(())
    }

    fn record(&self, record: &Record) -> Result<(), Error> {
        let mut locals = BTreeSet::new();
        if let Some(signature) = &record.signature {
            for generic in &signature.generics {
                locals.insert(generic.name.clone());
            }
            let type_locals = locals.clone();
            for param in &signature.params {
                locals.insert(param.name.clone());
            }
            if let Results::Fixed(results) = &signature.results {
                for result in results {
                    if let Some(name) = &result.name {
                        locals.insert(name.clone());
                    }
                }
            }
            for generic in &signature.generics {
                self.node(&generic.ty, Some(Space::Type), &type_locals)?;
            }
            for param in &signature.params {
                let space = if record.kind == "op"
                    && !self
                        .defs
                        .ops
                        .iter()
                        .find(|op| op.name == record.name)
                        .expect("checked operation")
                        .params
                        .iter()
                        .find(|p| p.name == param.name)
                        .is_some_and(|p| matches!(p.kind, crate::model::ParamKind::Property(_)))
                {
                    Space::Type
                } else {
                    Space::Data
                };
                self.node(&param.ty, Some(space), &type_locals)?;
            }
            if let Results::Fixed(results) = &signature.results {
                let space = if record.kind == "op" {
                    Space::Type
                } else {
                    Space::Data
                };
                for result in results {
                    self.node(&result.ty, Some(space), &type_locals)?;
                }
            }
        }
        if record.kind == "property" {
            self.name(&record.name, Space::Data, record.offset, &locals)?;
            locals.insert("value".into());
        }
        if let Some(FunctionBody::Value(body)) = &record.body {
            self.node(body, None, &locals)?;
        }
        match record.kind.as_str() {
            "struct" | "interface" => {
                for ty in record.fields.values() {
                    self.node(ty, Some(Space::Data), &locals)?;
                }
            }
            "type" | "typeset" | "predicate" => {
                for (name, value) in &record.fields {
                    let space = if name == "field" {
                        Space::Data
                    } else {
                        Space::Type
                    };
                    self.node(value, Some(space), &locals)?;
                }
            }
            "enum" => {
                if let Some(Node {
                    kind: Kind::List(variants),
                    ..
                }) = record.fields.get("variants")
                {
                    for variant in variants {
                        if let Kind::Call(_, args) = &variant.kind {
                            for ty in args {
                                self.node(ty, Some(Space::Data), &locals)?;
                            }
                        }
                    }
                }
            }
            "encoding" | "comparison" => {}
            _ => {
                for value in record.fields.values() {
                    self.node(value, None, &locals)?;
                }
            }
        }
        Ok(())
    }
}
