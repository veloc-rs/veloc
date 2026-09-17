//! Import visibility is lexical to each file, independently of loading order.
//! Resolution within a namespace remains the checked model's responsibility.
use std::collections::{BTreeMap, BTreeSet};

use crate::syntax::{Decl, DeclKind, FunctionBody, Kind, Node, Results};
use crate::{Error, model::records::rust_binding};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Space {
    Data,
    Type,
    Function,
    Value,
}

pub(crate) fn check(
    source: &str,
    declarations: &[Decl],
    files: &[super::File],
    data: &crate::model::data::Types,
) -> Result<(), Error> {
    let mut symbols = BTreeMap::<(Space, String), BTreeSet<usize>>::new();
    for (file, entry) in files.iter().enumerate() {
        for (owner, record) in crate::syntax::walk(&declarations[entry.declarations.clone()]) {
            let space = match &record.kind {
                DeclKind::Type { .. } if rust_binding(record).is_some() => Space::Data,
                DeclKind::Type { .. } | DeclKind::TypeSet(_) => Space::Type,
                DeclKind::Function { .. } => Space::Function,
                DeclKind::Constant { .. } => Space::Value,
                DeclKind::Fields(kind)
                    if matches!(kind.as_str(), "struct" | "enum" | "encoding") =>
                {
                    Space::Data
                }
                _ => continue,
            };
            symbols
                .entry((
                    space,
                    owner.map_or_else(
                        || record.name.clone(),
                        |owner| format!("{owner}::{}", record.name),
                    ),
                ))
                .or_default()
                .insert(file);
            if space == Space::Type && matches!(&record.kind, DeclKind::Type { .. }) {
                symbols
                    .entry((Space::Type, format!("Type::{}", record.name)))
                    .or_default()
                    .insert(file);
            }
            if let Some(en) = data.enums.iter().find(|en| en.name == record.name) {
                for (name, _) in &en.variants {
                    symbols
                        .entry((Space::Value, name.clone()))
                        .or_default()
                        .insert(file);
                }
            }
        }
    }
    for file in files {
        let checker = Checker {
            source,
            data,
            symbols: &symbols,
            files,
        };
        for (_, record) in crate::syntax::walk(&declarations[file.declarations.clone()]) {
            checker.record(record)?;
        }
    }
    Ok(())
}

struct Checker<'a> {
    data: &'a crate::model::data::Types,
    source: &'a str,
    symbols: &'a BTreeMap<(Space, String), BTreeSet<usize>>,
    files: &'a [super::File],
}

impl Checker<'_> {
    fn name(
        &self,
        name: &str,
        space: Space,
        offset: usize,
        locals: &BTreeSet<String>,
    ) -> Result<(), Error> {
        // Template nodes keep definition locations; substituted arguments keep
        // invocation locations. Resolve each reference in its lexical file.
        let owner = self
            .files
            .partition_point(|file| file.offset <= offset)
            .saturating_sub(1);
        let visible = &self.files[owner].visible;
        if !locals.contains(name)
            && let Some(owners) = self.symbols.get(&(space, name.to_owned()))
            && owners.is_disjoint(visible)
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
                // SSA reference constructors are declared data types whose
                // argument is a logical IR type, independently of storage.
                if self.data.rust.policy(name).references.is_operand() {
                    self.name(name, Space::Data, node.offset, locals)?;
                    for arg in args {
                        self.node(arg, Some(Space::Type), locals)?;
                    }
                    return Ok(());
                }
                if let Some((owner, _)) = name.split_once("::") {
                    self.name(owner, Space::Data, node.offset, locals)?;
                }
                self.name(name, space.unwrap_or(Space::Function), node.offset, locals)?;
                if space.is_none() {
                    self.name(name, Space::Value, node.offset, locals)?;
                }
                for arg in args {
                    self.node(arg, space, locals)?;
                }
            }
            Kind::TypedCall(name, types, args) => {
                self.name(name, Space::Value, node.offset, locals)?;
                for ty in types {
                    self.node(ty, Some(Space::Type), locals)?;
                }
                for arg in args {
                    self.node(arg, None, locals)?;
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
            Kind::Record(fields) => {
                for value in fields.values() {
                    self.node(value, None, locals)?;
                }
            }
            Kind::Scoped(param, body) => {
                self.node(&param.ty, Some(Space::Data), locals)?;
                let mut locals = locals.clone();
                locals.insert(param.name.clone());
                self.node(body, space, &locals)?;
            }
            Kind::Query(_, value) | Kind::Let(_, value) => self.node(value, None, locals)?,
            Kind::List(nodes) | Kind::Union(nodes) | Kind::Intersection(nodes) => {
                let mut locals = locals.clone();
                for node in nodes {
                    self.node(node, space, &locals)?;
                    if let Kind::Let(name, _) = &node.kind {
                        locals.insert(name.clone());
                    }
                }
            }
            Kind::Unary(_, value) | Kind::Try(value) | Kind::Ref(value) => {
                self.node(value, space, locals)?
            }
            Kind::Binary(_, a, b) => {
                self.node(a, None, locals)?;
                self.node(b, None, locals)?;
            }
            Kind::Lambda(names, body) => {
                let mut locals = locals.clone();
                locals.extend(names.iter().cloned());
                self.node(body, None, &locals)?;
            }
            Kind::Match(value, arms) => {
                self.node(value, None, locals)?;
                for arm in arms {
                    if !matches!(&arm.pattern.kind, Kind::Name(name) if name == "_") {
                        self.node(&arm.pattern, None, locals)?;
                    }
                    if let Some(guard) = &arm.guard {
                        self.node(guard, None, locals)?;
                    }
                    self.node(&arm.value, None, locals)?;
                }
            }
            Kind::Text(_) | Kind::Number(_) | Kind::Integer(_) => {}
        }
        Ok(())
    }

    fn record(&self, record: &Decl) -> Result<(), Error> {
        let mut locals = BTreeSet::new();
        if let Some(signature) = record.signature() {
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
                self.node(&param.ty, Some(Space::Data), &type_locals)?;
            }
            if let Results::Fixed(results) = &signature.results {
                let space = if matches!(&record.kind, DeclKind::Op(_)) {
                    Space::Type
                } else {
                    Space::Data
                };
                for result in results {
                    self.node(&result.ty, Some(space), &type_locals)?;
                }
            }
        }
        if let Some(FunctionBody::Value(body)) = record.body() {
            self.node(body, None, &locals)?;
        }
        match &record.kind {
            DeclKind::Type { binding, .. } | DeclKind::TypeSet(binding) => {
                self.node(binding, Some(Space::Type), &locals)?;
            }
            DeclKind::Constant { ty, value } => {
                self.node(ty, Some(Space::Data), &locals)?;
                if let Some(value) = value {
                    self.node(value, None, &locals)?;
                }
            }
            _ => {}
        }
        match &record.kind {
            DeclKind::Fields(kind) if kind == "struct" => {
                for ty in record.fields.values() {
                    self.node(ty, Some(Space::Data), &locals)?;
                }
            }
            DeclKind::Type { .. } | DeclKind::TypeSet(_) => {
                for (name, value) in &record.fields {
                    let space = if name == "field" {
                        Space::Data
                    } else {
                        Space::Type
                    };
                    self.node(value, Some(space), &locals)?;
                }
            }
            DeclKind::Fields(kind) if kind == "enum" => {
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
            DeclKind::Fields(kind) if kind == "encoding" => {}
            _ => {
                for value in record.fields.values() {
                    self.node(value, None, &locals)?;
                }
            }
        }
        Ok(())
    }
}
