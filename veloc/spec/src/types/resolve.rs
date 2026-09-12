//! Resolve named compositions over the shared Rust type catalog.
use crate::types::{Scalar, TypeSet};
use crate::{
    Error,
    model::Fields,
    syntax::{Kind, Node, Record},
};
use std::collections::BTreeMap;
use veloc_types::{ScalarType, Type};

pub(crate) struct Declarations {
    pub scalars: Vec<Scalar>,
    pub exact: BTreeMap<String, TypeSet>,
}

// Keep the receiver in the AST so file-local import checks still see Type.
pub(crate) fn name(node: &Node) -> Option<String> {
    match &node.kind {
        Kind::Name(name) => Some(name.clone()),
        Kind::Member(receiver, member) if matches!(&receiver.kind, Kind::Name(name) if name == "Type") => {
            Some(format!("Type.{member}"))
        }
        _ => None,
    }
}

pub(crate) fn compile(records: &[Record], source: &str) -> Result<Declarations, Error> {
    let bound = records.iter().any(|r| {
        r.kind == "type" && r.name == "Type" && crate::model::records::rust_binding(r).is_some()
    });
    let mut resolved = if bound {
        Type::NAMED
            .iter()
            .map(|&(name, ty)| (format!("Type.{name}"), ty))
            .collect()
    } else {
        BTreeMap::new()
    };
    let scalars = if bound {
        ScalarType::ALL
            .iter()
            .map(|&scalar| Scalar {
                name: format!("{:?}", scalar.as_type()),
                ty: scalar.element(),
            })
            .collect()
    } else {
        Vec::new()
    };
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
        pending.insert(record.name.clone(), expr);
    }
    while !pending.is_empty() {
        let before = pending.len();
        for alias in pending.keys().cloned().collect::<Vec<_>>() {
            if let Some(ty) = resolve(source, &pending[&alias], &resolved, &pending)? {
                pending.remove(&alias);
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
        .map(|(name, ty)| {
            let (exponent, scalable) = ty
                .as_vector()
                .map(|v| {
                    let (lanes, scalable) = v.shape();
                    (lanes.trailing_zeros(), scalable)
                })
                .unwrap_or((0, false));
            (
                name,
                TypeSet::singleton(ty.element().expect("compact type"), exponent, scalable),
            )
        })
        .collect();
    Ok(Declarations { scalars, exact })
}

fn resolve(
    source: &str,
    node: &Node,
    resolved: &BTreeMap<String, Type>,
    pending: &BTreeMap<String, Node>,
) -> Result<Option<Type>, Error> {
    let fail = |message| Error::at(source, node.offset, message);
    if let Some(name) = name(node) {
        if let Some(&ty) = resolved.get(&name) {
            return Ok(Some(ty));
        }
        if pending.contains_key(&name) {
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
    let scalar = element
        .as_scalar()
        .ok_or_else(|| fail("vector element must be a scalar type".into()))?;
    let lanes = u16::try_from(lanes)
        .map_err(|_| fail("vector lanes exceed the supported type domain".into()))?;
    scalar
        .vector(lanes, scalable)
        .map(|v| Some(v.as_type()))
        .ok_or_else(|| fail("invalid vector lanes or element type".into()))
}
