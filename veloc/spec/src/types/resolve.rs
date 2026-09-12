//! Resolve type construction independently of the compact MIR representation.

use std::collections::BTreeMap;

use crate::Error;
use crate::model::Fields;
use crate::syntax::{Kind, Node, Record};
use crate::types::TypeSet;
use crate::types::{Primitive, Scalar};

fn primitive_name(ty: Primitive) -> String {
    match ty {
        Primitive::Int(bits) => format!("I{bits}"),
        Primitive::Float(bits) => format!("F{bits}"),
        Primitive::Bool => "Bool".into(),
        Primitive::Ptr => "Ptr".into(),
    }
}

#[derive(Clone, Copy)]
enum TypeExpr {
    Scalar(Primitive),
    Vector {
        element: Primitive,
        lanes: u32,
        scalable: bool,
    },
}

pub(crate) struct Declarations {
    pub scalars: Vec<Scalar>,
    pub exact: BTreeMap<String, TypeSet>,
}

pub(crate) fn compile(records: &[Record], source: &str) -> Result<Declarations, Error> {
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
    let mut resolved = BTreeMap::new();
    while !pending.is_empty() {
        let before = pending.len();
        for name in pending.keys().cloned().collect::<Vec<_>>() {
            if let Some(ty) = resolve(source, &pending[&name], &resolved, &pending)? {
                let node = pending.remove(&name).unwrap();
                resolved.insert(name, (node.offset, ty));
            }
        }
        if pending.len() == before {
            let (name, node) = pending.first_key_value().unwrap();
            return Err(Error::at(
                source,
                node.offset,
                format!("cyclic type definition `{name}`"),
            ));
        }
    }

    // Canonical logical scalar kinds, independent of backend bit encodings.
    let primitives = resolved
        .values()
        .map(|(_, ty)| match ty {
            TypeExpr::Scalar(p) => *p,
            TypeExpr::Vector { element, .. } => *element,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let scalars = primitives
        .into_iter()
        .map(|ty| Scalar {
            name: primitive_name(ty),
            ty,
        })
        .collect::<Vec<_>>();
    let mut exact = BTreeMap::new();
    for (name, (offset, ty)) in resolved {
        let primitive = match ty {
            TypeExpr::Scalar(p) => p,
            TypeExpr::Vector { element, .. } => element,
        };
        let set = match ty {
            TypeExpr::Scalar(_) => TypeSet::singleton(primitive, 0, false),
            TypeExpr::Vector {
                lanes, scalable, ..
            } => {
                if lanes > u32::from(veloc_types::MAX_VECTOR_LANES) {
                    return Err(Error::at(
                        source,
                        offset,
                        "vector lanes exceed the supported type domain",
                    ));
                }
                TypeSet::singleton(primitive, lanes.trailing_zeros(), scalable)
            }
        };
        exact.insert(name, set);
    }
    Ok(Declarations { scalars, exact })
}

fn resolve(
    source: &str,
    node: &Node,
    resolved: &BTreeMap<String, (usize, TypeExpr)>,
    pending: &BTreeMap<String, Node>,
) -> Result<Option<TypeExpr>, Error> {
    let fail = |message| Error::at(source, node.offset, message);
    if let Kind::Name(name) = &node.kind {
        if let Some((_, ty)) = resolved.get(name) {
            return Ok(Some(*ty));
        }
        if pending.contains_key(name) {
            return Ok(None);
        }
        return Err(fail(format!("unknown type `{name}`")));
    }
    let Kind::Call(constructor, args) = &node.kind else {
        return Err(fail("expected a type name or type constructor".into()));
    };
    let primitive = match constructor.as_str() {
        "int" | "float" => {
            let [bits] = args.as_slice() else {
                return Err(fail(format!("{constructor} expects one bit width")));
            };
            let bits = number(source, bits)?;
            match (constructor.as_str(), bits) {
                ("int", 8 | 16 | 32 | 64) => Primitive::Int(bits),
                ("float", 32 | 64) => Primitive::Float(bits),
                _ => {
                    return Err(fail(
                        "unsupported scalar kind or width for the MIR codecs".into(),
                    ));
                }
            }
        }
        "bool" | "ptr" => {
            if !args.is_empty() {
                return Err(fail(format!("{constructor} expects no arguments")));
            }
            if constructor == "bool" {
                Primitive::Bool
            } else {
                Primitive::Ptr
            }
        }
        "vector" => {
            let [element, shape] = args.as_slice() else {
                return Err(fail("vector expects an element type and lane count".into()));
            };
            let (lanes, scalable) = match &shape.kind {
                Kind::Number(n) => (*n, false),
                Kind::Call(name, args) if name == "scalable" && args.len() == 1 => {
                    (number(source, &args[0])?, true)
                }
                _ => {
                    return Err(Error::at(
                        source,
                        shape.offset,
                        "vector shape must be a lane count or scalable(lanes)",
                    ));
                }
            };
            if lanes < 2 || !lanes.is_power_of_two() {
                return Err(fail(
                    "vector lanes must be a power of two and at least two".into(),
                ));
            }
            let Some(element) = resolve(source, element, resolved, pending)? else {
                return Ok(None);
            };
            let TypeExpr::Scalar(element) = element else {
                return Err(fail("vector element must be a scalar type".into()));
            };
            if element == Primitive::Ptr {
                return Err(fail("pointer vectors are not supported".into()));
            }
            return Ok(Some(TypeExpr::Vector {
                element,
                lanes,
                scalable,
            }));
        }
        _ => return Err(fail(format!("unknown type constructor `{constructor}`"))),
    };
    Ok(Some(TypeExpr::Scalar(primitive)))
}

fn number(source: &str, node: &Node) -> Result<u32, Error> {
    match node.kind {
        Kind::Number(n) => Ok(n),
        _ => Err(Error::at(source, node.offset, "expected a number")),
    }
}
