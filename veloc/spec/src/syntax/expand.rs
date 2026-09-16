//! Definition-time templates expand AST nodes, never source strings. Node
//! locations survive substitution so import lookup retains lexical ownership.
use std::collections::{BTreeMap, BTreeSet};

use super::{Decl, DeclKind, FunctionBody, Kind, Node, Results, Signature};
use crate::Error;

/// Keep one output group per input declaration, allowing the loader to retain
/// file ownership even when a template produces several declarations.
pub(crate) fn expand(
    source: &str,
    declarations: &[Decl],
    visible: impl Fn(usize, usize) -> bool,
) -> Result<Vec<Vec<Decl>>, Error> {
    let mut templates = BTreeMap::new();
    for declaration in declarations {
        let DeclKind::Template { params, .. } = &declaration.kind else {
            continue;
        };
        if templates
            .insert(declaration.name.as_str(), declaration)
            .is_some()
        {
            return Err(Error::at(source, declaration.offset, "duplicate template"));
        }
        let mut names = BTreeSet::new();
        for parameter in params {
            if parameter.moves
                || !matches!(&parameter.ty.kind, Kind::Name(kind) if matches!(kind.as_str(), "ident" | "type" | "expr"))
            {
                return Err(Error::at(
                    source,
                    parameter.offset,
                    "template parameter requires ident, type, or expr",
                ));
            }
            if !names.insert(&parameter.name) {
                return Err(Error::at(
                    source,
                    parameter.offset,
                    "duplicate template parameter",
                ));
            }
        }
    }
    let context = Expansion {
        source,
        templates,
        visible: &visible,
    };
    declarations
        .iter()
        .map(|decl| context.declaration(decl, &mut Vec::new()))
        .collect()
}

struct Expansion<'a, F> {
    source: &'a str,
    templates: BTreeMap<&'a str, &'a Decl>,
    visible: &'a F,
}

impl<F: Fn(usize, usize) -> bool> Expansion<'_, F> {
    fn declaration(&self, decl: &Decl, active: &mut Vec<String>) -> Result<Vec<Decl>, Error> {
        let args = match &decl.kind {
            DeclKind::Template { .. } => return Ok(Vec::new()),
            DeclKind::Expand(args) => args,
            _ => return Ok(vec![decl.clone()]),
        };
        let fail = |message| Error::at(self.source, decl.offset, message);
        let template = self
            .templates
            .get(decl.name.as_str())
            .ok_or_else(|| fail(format!("unknown template `{}`", decl.name)))?;
        if !(self.visible)(decl.offset, template.offset) {
            return Err(fail(format!("template `{}` is not imported", decl.name)));
        }
        if active.contains(&decl.name) || active.len() >= 128 {
            return Err(fail(format!(
                "recursive or excessive template expansion: {} -> {}",
                active.join(" -> "),
                decl.name
            )));
        }
        let DeclKind::Template { params, body } = &template.kind else {
            unreachable!()
        };
        if params.len() != args.len() {
            return Err(fail(format!(
                "template `{}` expects {} arguments, got {}",
                decl.name,
                params.len(),
                args.len()
            )));
        }
        let bindings: BTreeMap<_, _> = params
            .iter()
            .zip(args)
            .map(|(param, arg)| {
                if matches!(&param.ty.kind, Kind::Name(kind) if kind == "ident")
                    && !matches!(arg.kind, Kind::Name(_))
                {
                    return Err(Error::at(
                        self.source,
                        arg.offset,
                        "ident template argument requires a name",
                    ));
                }
                Ok((param.name.clone(), arg.clone()))
            })
            .collect::<Result<_, _>>()?;
        active.push(decl.name.clone());
        let mut result = Vec::new();
        for item in body {
            let mut item = item.clone();
            substitute_decl(self.source, &mut item, &bindings)?;
            let expanded = self.declaration(&item, active).map_err(|mut error| {
                let site = Error::at(self.source, decl.offset, "");
                error.message.push_str(&format!(
                    "\n  expanded from {} at {}:{}",
                    decl.name, site.line, site.column
                ));
                error
            })?;
            result.extend(expanded);
        }
        active.pop();
        Ok(result)
    }
}

fn substitute_name(
    source: &str,
    name: &mut String,
    offset: usize,
    bindings: &BTreeMap<String, Node>,
) -> Result<(), Error> {
    if let Some(arg) = bindings.get(name) {
        let Kind::Name(value) = &arg.kind else {
            return Err(Error::at(
                source,
                offset,
                "template substitution in a name requires an identifier",
            ));
        };
        *name = value.clone();
    }
    Ok(())
}

fn signature(
    source: &str,
    signature: &mut Signature,
    bindings: &BTreeMap<String, Node>,
) -> Result<(), Error> {
    for param in signature.generics.iter_mut().chain(&mut signature.params) {
        reject_shadow(source, &param.name, param.offset, bindings)?;
        node(source, &mut param.ty, bindings)?;
    }
    if let Results::Fixed(results) = &mut signature.results {
        for result in results {
            if let Some(name) = &result.name {
                reject_shadow(source, name, result.offset, bindings)?;
            }
            node(source, &mut result.ty, bindings)?;
        }
    }
    Ok(())
}

// Public operand names must remain stable for encodings and storage mappings.
// Reject capture rather than silently alpha-renaming these interface names.
fn reject_shadow(
    source: &str,
    name: &str,
    offset: usize,
    bindings: &BTreeMap<String, Node>,
) -> Result<(), Error> {
    if bindings.contains_key(name) {
        return Err(Error::at(
            source,
            offset,
            format!("local `{name}` shadows a template parameter"),
        ));
    }
    for value in bindings.values() {
        let mut found = false;
        let mut value = value.clone();
        visit_names(&mut value, &mut |candidate| {
            found |= candidate == name;
        });
        if found {
            return Err(Error::at(
                source,
                offset,
                format!("local `{name}` would capture a template argument"),
            ));
        }
    }
    Ok(())
}

fn substitute_decl(
    source: &str,
    decl: &mut Decl,
    bindings: &BTreeMap<String, Node>,
) -> Result<(), Error> {
    substitute_name(source, &mut decl.name, decl.offset, bindings)?;
    for value in decl.fields.values_mut() {
        node(source, value, bindings)?;
    }
    match &mut decl.kind {
        DeclKind::Op(sig) => signature(source, sig, bindings)?,
        DeclKind::Function {
            signature: sig,
            body,
        } => {
            signature(source, sig, bindings)?;
            if let FunctionBody::Value(value) = body {
                node(source, value, bindings)?;
            }
        }
        DeclKind::Type { binding, members } => {
            node(source, binding, bindings)?;
            for member in members {
                substitute_decl(source, member, bindings)?;
            }
        }
        DeclKind::TypeSet(value) => node(source, value, bindings)?,
        DeclKind::Constant { ty, value } => {
            node(source, ty, bindings)?;
            if let Some(value) = value {
                node(source, value, bindings)?;
            }
        }
        DeclKind::Expand(args) => {
            for arg in args {
                node(source, arg, bindings)?;
            }
        }
        DeclKind::Fields(_) => {}
        DeclKind::Template { .. } => unreachable!("nested templates rejected by parser"),
    }
    Ok(())
}

fn node(source: &str, value: &mut Node, bindings: &BTreeMap<String, Node>) -> Result<(), Error> {
    if let Kind::Name(name) = &value.kind
        && let Some(arg) = bindings.get(name)
    {
        *value = arg.clone();
        return Ok(());
    }
    match &mut value.kind {
        Kind::Call(name, _) | Kind::Object(name, _) => {
            substitute_name(source, name, value.offset, bindings)?
        }
        Kind::Lambda(params, _) => {
            for param in params {
                reject_shadow(source, param, value.offset, bindings)?;
            }
        }
        Kind::Let(name, _) => reject_shadow(source, name, value.offset, bindings)?,
        Kind::Scoped(param, _) => reject_shadow(source, &param.name, param.offset, bindings)?,
        _ => {}
    }
    children(value, &mut |child| node(source, child, bindings))
}

fn visit_names(node: &mut Node, visit: &mut impl FnMut(&str)) {
    if let Kind::Name(name) = &node.kind {
        visit(name);
    }
    let _: Result<(), core::convert::Infallible> = children(node, &mut |child| {
        visit_names(child, visit);
        Ok(())
    });
}

fn children<E>(
    node: &mut Node,
    visit: &mut impl FnMut(&mut Node) -> Result<(), E>,
) -> Result<(), E> {
    match &mut node.kind {
        Kind::List(items)
        | Kind::Call(_, items)
        | Kind::Union(items)
        | Kind::Intersection(items) => {
            for item in items {
                visit(item)?;
            }
        }
        Kind::Object(_, fields) | Kind::Record(fields) => {
            for field in fields.values_mut() {
                visit(field)?;
            }
        }
        Kind::Method(receiver, _, args) => {
            visit(receiver)?;
            for arg in args {
                visit(arg)?;
            }
        }
        Kind::Binary(_, lhs, rhs) => {
            visit(lhs)?;
            visit(rhs)?;
        }
        Kind::Member(value, _)
        | Kind::Unary(_, value)
        | Kind::Lambda(_, value)
        | Kind::Try(value)
        | Kind::Ref(value)
        | Kind::Let(_, value)
        | Kind::Query(_, value) => visit(value)?,
        Kind::Scoped(param, value) => {
            visit(&mut param.ty)?;
            visit(value)?;
        }
        Kind::Name(_) | Kind::Text(_) | Kind::Number(_) | Kind::Integer(_) => {}
    }
    Ok(())
}
