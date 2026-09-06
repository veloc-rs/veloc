//! Resolve type construction independently of the compact MIR representation.

use std::collections::BTreeMap;

use crate::Error;
use crate::encoding::TypeEncoding;
use crate::model::Fields;
use crate::syntax::{Kind, Node, Record};
use crate::type_set::TypeSet;
use crate::types::{Scalar, ScalarKind, Vector};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Primitive {
    kind: ScalarKind,
    bits: Option<u32>,
}

impl Primitive {
    fn name(self) -> String {
        match self.kind {
            ScalarKind::Integer => format!("I{}", self.bits.unwrap()),
            ScalarKind::Float => format!("F{}", self.bits.unwrap()),
            ScalarKind::Boolean => "Bool".into(),
            ScalarKind::Pointer => "Ptr".into(),
        }
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
    pub vectors: Vec<Vector>,
    pub aliases: Vec<(String, usize)>,
    pub exact: BTreeMap<String, TypeSet>,
}

pub(crate) fn compile(
    records: &[Record],
    source: &str,
    encoding: &TypeEncoding,
) -> Result<Declarations, Error> {
    let mut pending = BTreeMap::new();
    for record in records.iter().filter(|r| r.kind == "type") {
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

    // Bind structural scalar types to the explicit backend encoding. A primitive
    // has one code; alternate type names are aliases, not new representations.
    let mut scalars = Vec::new();
    let mut indices = BTreeMap::new();
    let mut codes = encoding.codes.iter().collect::<Vec<_>>();
    codes.sort_by_key(|(_, entry)| entry.code);
    for (name, entry) in codes {
        let (_, ty) = resolved.get(name).ok_or_else(|| {
            Error::at(
                source,
                entry.offset,
                format!("encoding references unknown type `{name}`"),
            )
        })?;
        let TypeExpr::Scalar(primitive) = *ty else {
            return Err(Error::at(
                source,
                entry.offset,
                "scalar encoding cannot assign a code to a vector",
            ));
        };
        let canonical = primitive.name();
        if *name != canonical.to_ascii_uppercase() {
            return Err(Error::at(
                source,
                entry.offset,
                format!(
                    "MIR scalar encoding requires name `{}` for this primitive",
                    canonical.to_ascii_uppercase()
                ),
            ));
        }
        indices.insert(primitive, scalars.len());
        scalars.push(Scalar {
            name: canonical,
            code: entry.code,
            kind: primitive.kind,
            bits: primitive.bits,
        });
    }
    let mut exact = BTreeMap::new();
    let mut aliases = Vec::new();
    let mut vectors = Vec::new();
    for (name, (offset, ty)) in resolved {
        let primitive = match ty {
            TypeExpr::Scalar(p) => p,
            TypeExpr::Vector { element, .. } => element,
        };
        let index = *indices.get(&primitive).ok_or_else(|| {
            Error::at(
                source,
                offset,
                format!(
                    "missing scalar encoding for `{}`",
                    primitive.name().to_ascii_uppercase()
                ),
            )
        })?;
        let scalar = &scalars[index];
        let set = match ty {
            TypeExpr::Scalar(_) => {
                if name != scalar.exact() {
                    aliases.push((name.clone(), index));
                }
                TypeSet::singleton(scalar.code, 0, false)
            }
            TypeExpr::Vector {
                lanes, scalable, ..
            } => {
                if lanes.trailing_zeros() > encoding.lanes_log2_max() {
                    return Err(Error::at(
                        source,
                        offset,
                        "vector lanes must fit Type encoding",
                    ));
                }
                vectors.push(Vector {
                    name: name.clone(),
                    element: index,
                    lanes: lanes as u16,
                    scalable,
                });
                TypeSet::singleton(scalar.code, lanes.trailing_zeros(), scalable)
            }
        };
        exact.insert(name, set);
    }
    Ok(Declarations {
        scalars,
        vectors,
        aliases,
        exact,
    })
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
            let kind = match (constructor.as_str(), bits) {
                ("int", 8 | 16 | 32 | 64) => ScalarKind::Integer,
                ("float", 32 | 64) => ScalarKind::Float,
                _ => {
                    return Err(fail(
                        "unsupported scalar kind or width for the MIR codecs".into(),
                    ));
                }
            };
            Primitive {
                kind,
                bits: Some(bits),
            }
        }
        "bool" | "ptr" => {
            if !args.is_empty() {
                return Err(fail(format!("{constructor} expects no arguments")));
            }
            if constructor == "bool" {
                Primitive {
                    kind: ScalarKind::Boolean,
                    bits: Some(1),
                }
            } else {
                Primitive {
                    kind: ScalarKind::Pointer,
                    bits: None,
                }
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
            if element.kind == ScalarKind::Pointer {
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
