//! Resolve logical scalar domains, aliases and vector compositions.
use super::{Primitive as Element, TypeKey};
use crate::types::{Scalar, TypeSet};
use crate::{
    Error,
    model::Fields,
    syntax::{Kind, Node, Record},
};
use std::collections::BTreeMap;

pub(crate) struct Declarations {
    pub scalars: Vec<Scalar>,
    pub exact: BTreeMap<String, TypeSet>,
}

pub(crate) fn name(node: &Node) -> Option<String> {
    match &node.kind {
        Kind::Name(name) => Some(name.clone()),
        _ => None,
    }
}

pub(crate) fn compile(records: &[Record], source: &str) -> Result<Declarations, Error> {
    let mut resolved = BTreeMap::new();
    let mut scalars = Vec::new();
    for record in records
        .iter()
        .filter(|r| r.kind == "type" && crate::model::records::rust_binding(r).is_none())
    {
        let node = &record.fields["expr"];
        let scalar = match &node.kind {
            Kind::Name(name) if name == "bool" => Some(Element::Bool),
            Kind::Name(name) if name == "ptr" => Some(Element::Ptr),
            Kind::Call(name, args) if matches!(name.as_str(), "int" | "float") => {
                let [
                    Node {
                        kind: Kind::Number(bits),
                        ..
                    },
                ] = args.as_slice()
                else {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "scalar type expects a bit width",
                    ));
                };
                if *bits == 0 || *bits > 128 {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "scalar width must be in 1..=128",
                    ));
                }
                Some(if name == "int" {
                    Element::Int(*bits)
                } else {
                    Element::Float(*bits)
                })
            }
            _ => None,
        };
        if let Some(ty) = scalar {
            if scalars.iter().any(|scalar: &Scalar| scalar.ty == ty) {
                return Err(Error::at(
                    source,
                    node.offset,
                    "duplicate scalar domain; use a type alias",
                ));
            }
            scalars.push(Scalar {
                name: record.name.clone(),
                ty,
            });
            resolved.insert(record.name.clone(), (ty, 0));
            resolved.insert(format!("Type::{}", record.name), (ty, 0));
        }
    }
    let mut pending = BTreeMap::new();
    for record in records
        .iter()
        .filter(|r| r.kind == "type" && crate::model::records::rust_binding(r).is_none())
    {
        let mut fields = Fields::new(source, record.clone());
        if record.name != record.name.to_ascii_uppercase() || record.name == "INVALID" {
            return Err(fields.error("type name must be uppercase and not INVALID"));
        }
        let expr = fields.take("expr")?;
        fields.finish()?;
        if !resolved.contains_key(&record.name) {
            pending.insert(record.name.clone(), expr);
        }
    }
    while !pending.is_empty() {
        let before = pending.len();
        for alias in pending.keys().cloned().collect::<Vec<_>>() {
            if let Some(ty) = resolve(source, &pending[&alias], &resolved, &pending)? {
                pending.remove(&alias);
                resolved.insert(format!("Type::{alias}"), ty);
                resolved.insert(alias, ty);
            }
        }
        if pending.len() == before {
            let (alias, node) = pending.first_key_value().unwrap();
            return Err(Error::at(
                source,
                node.offset,
                format!("cyclic type definition `{alias}`"),
            ));
        }
    }
    let exact = resolved
        .into_iter()
        .map(|(name, (element, shape))| {
            (name, TypeSet::singleton(element, shape % 16, shape >= 16))
        })
        .collect();
    Ok(Declarations { scalars, exact })
}

fn resolve(
    source: &str,
    node: &Node,
    resolved: &BTreeMap<String, TypeKey>,
    pending: &BTreeMap<String, Node>,
) -> Result<Option<TypeKey>, Error> {
    let fail = |message| Error::at(source, node.offset, message);
    if let Some(name) = name(node) {
        if let Some(&ty) = resolved.get(&name) {
            return Ok(Some(ty));
        }
        if pending.contains_key(name.strip_prefix("Type::").unwrap_or(&name)) {
            return Ok(None);
        }
        return Err(fail(format!("unknown type `{name}`")));
    }
    let Kind::Call(constructor, args) = &node.kind else {
        return Err(fail("expected a type name or type constructor".into()));
    };
    if constructor != "vector" {
        return Err(fail(format!(
            "unknown type constructor `{constructor}`; use Type constants for scalar types"
        )));
    }
    let [element, shape] = args.as_slice() else {
        return Err(fail("vector expects an element type and lane count".into()));
    };
    let number = |node: &Node| match node.kind {
        Kind::Number(n) => Ok(n),
        _ => Err(Error::at(source, node.offset, "expected a number")),
    };
    let (lanes, scalable) = match &shape.kind {
        Kind::Number(n) => (*n, false),
        Kind::Call(name, args) if name == "scalable" && args.len() == 1 => {
            (number(&args[0])?, true)
        }
        _ => {
            return Err(fail(
                "vector shape must be a lane count or scalable(lanes)".into(),
            ));
        }
    };
    let Some(element) = resolve(source, element, resolved, pending)? else {
        return Ok(None);
    };
    if element.1 != 0 {
        return Err(fail("vector element must be a scalar type".into()));
    }
    let lanes = u16::try_from(lanes)
        .map_err(|_| fail("vector lanes exceed the supported type domain".into()))?;
    if !lanes.is_power_of_two() || lanes <= 1 {
        return Err(fail("invalid vector lanes or element type".into()));
    }
    let key = (
        element.0,
        lanes.trailing_zeros() + if scalable { 16 } else { 0 },
    );
    if key.0 == Element::Ptr {
        return Err(fail("invalid vector lanes or element type".into()));
    }
    Ok(Some(key))
}
