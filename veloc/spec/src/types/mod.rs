//! Scalar types, named vectors and exact type-set expressions from types.ops.

use std::collections::BTreeMap;

use crate::Error;
use crate::model::Fields;
use crate::syntax::{Decl, DeclKind, Kind, Node};

pub(crate) mod cases;
pub(crate) mod generate;
mod resolve;
pub(crate) mod rules;

/// Exact type sets: logical scalar kinds map to scalar/fixed/scalable shape masks.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeSet(pub(crate) BTreeMap<Primitive, u32>);

impl TypeSet {
    pub fn singleton(code: Primitive, exponent: u32, scalable: bool) -> Self {
        Self(BTreeMap::from([(
            code,
            1 << (exponent + if scalable { 16 } else { 0 }),
        )]))
    }

    pub fn is_singleton(&self) -> bool {
        self.0.len() == 1 && self.0.values().next().unwrap().count_ones() == 1
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn union(&mut self, other: &Self) {
        for (&code, &shapes) in &other.0 {
            *self.0.entry(code).or_default() |= shapes;
        }
    }

    pub fn subset_of(&self, other: &Self) -> bool {
        !self.is_empty()
            && self
                .0
                .iter()
                .all(|(code, shapes)| shapes & !other.0.get(code).copied().unwrap_or(0) == 0)
    }

    pub fn intersect(&mut self, other: &Self) {
        self.0.retain(|code, shapes| {
            *shapes &= other.0.get(code).copied().unwrap_or(0);
            *shapes != 0
        });
    }

    pub fn retain_shapes(&mut self, allowed: u32) {
        self.0.retain(|_, shapes| {
            *shapes &= allowed;
            *shapes != 0
        });
    }

    /// Caller has checked that all members are non-pointer scalars.
    pub fn vectors(&self, max_exponent: u32) -> Self {
        let fixed = ((1u32 << (max_exponent + 1)) - 1) & !1;
        let shapes = fixed | (fixed << 16);
        Self(self.0.keys().map(|&code| (code, shapes)).collect())
    }
}

/// Mathematical element domains used by type-set and bit-vector analysis.
/// This describes a domain, not a runtime type representation or its catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Primitive {
    Int(u32),
    Float(u32),
    Bool,
    Ptr,
}

impl Primitive {
    pub fn element_bits(self) -> Option<u32> {
        match self {
            Self::Int(n) | Self::Float(n) => Some(n),
            Self::Bool => Some(1),
            Self::Ptr => None,
        }
    }
}

/// An element and a logical shape index: zero is scalar; 1..16 fixed,
/// 16..32 scalable. These indices only index the generator's type sets.
pub type TypeKey = (Primitive, u32);

#[derive(Debug)]
pub(crate) struct Scalar {
    pub name: String,
    pub ty: Primitive,
}

impl Scalar {
    pub fn exact(&self) -> String {
        self.name.to_ascii_uppercase()
    }
}

pub(crate) struct Types {
    pub scalars: Vec<Scalar>,
    pub exact: BTreeMap<String, TypeSet>,
    pub sets: BTreeMap<String, TypeSet>,
    pub lanes: TypeSet,
    pub integers: TypeSet,
    pub scalar_floats: TypeSet,
    max_exponent: u32,
}

pub(crate) fn rust_type(name: &str) -> String {
    format!(
        "crate::types::{}",
        name.strip_prefix("Type::").unwrap_or(name)
    )
}

impl Types {
    pub(crate) fn exact_name(&self, node: &Node) -> Option<String> {
        resolve::name(node).filter(|name| self.exact.contains_key(name))
    }
    /// Result domains supplied by typed literal properties, independent of set names.
    pub(crate) fn property_types(&self, name: &str) -> Option<TypeSet> {
        match name {
            "Float" => Some(self.scalar_floats.clone()),
            "Int" => {
                let mut set = self.integers.clone();
                set.retain_shapes(1);
                Some(set)
            }
            "VectorConst" => Some(self.lanes.vectors(self.max_exponent)),
            _ => None,
        }
    }

    pub fn compile(records: &[Decl], source: &str) -> Result<Self, Error> {
        if let Some(record) = records.iter().find(|r| {
            matches!(&r.kind, DeclKind::Type { .. } | DeclKind::TypeSet(_)) && r.name == "Callable"
        }) {
            return Err(Error::at(
                source,
                record.offset,
                "Callable is a reserved structural type name",
            ));
        }
        let declarations = crate::types::resolve::compile(records, source)?;
        let mut types = Self {
            scalars: declarations.scalars,
            exact: declarations.exact,
            sets: BTreeMap::new(),
            lanes: TypeSet::default(),
            integers: TypeSet::default(),
            scalar_floats: TypeSet::default(),
            max_exponent: 15,
        };
        for scalar in &types.scalars {
            let single = TypeSet::singleton(scalar.ty, 0, false);
            let mut family = single.clone();
            if scalar.ty != Primitive::Ptr {
                types.lanes.union(&single);
                family.union(&single.vectors(15));
            }
            if matches!(scalar.ty, Primitive::Int(_) | Primitive::Bool) {
                types.integers.union(&family);
            }
            if matches!(scalar.ty, Primitive::Float(_)) {
                types.scalar_floats.union(&single);
            }
        }
        let mut pending = BTreeMap::new();
        for record in records
            .iter()
            .filter(|r| matches!(&r.kind, DeclKind::TypeSet(_)))
        {
            let fields = Fields::new(source, record.clone());
            if types.exact.contains_key(&record.name) {
                return Err(fields.error("typeset name shadows an exact type"));
            }
            if matches!(
                record.name.as_str(),
                "values" | "successor" | "successors" | "signature"
            ) {
                return Err(fields.error("typeset name shadows a signature keyword"));
            }
            let crate::syntax::DeclKind::TypeSet(expr) = &record.kind else {
                unreachable!()
            };
            fields.finish()?;
            pending.insert(record.name.clone(), (record.offset, expr.clone()));
        }
        while !pending.is_empty() {
            let before = pending.len();
            for key in pending.keys().cloned().collect::<Vec<_>>() {
                let (offset, expr) = &pending[&key];
                if let Some(set) = types.member(source, expr, &pending)? {
                    if set.is_empty() {
                        return Err(Error::at(source, *offset, "type set must not be empty"));
                    }
                    pending.remove(&key);
                    types.sets.insert(key, set);
                }
            }
            if pending.len() == before {
                let (name, (offset, _)) = pending.first_key_value().unwrap();
                return Err(Error::at(
                    source,
                    *offset,
                    format!("cyclic type set `{name}`"),
                ));
            }
        }
        Ok(types)
    }

    fn member(
        &self,
        source: &str,
        node: &Node,
        pending: &BTreeMap<String, (usize, Node)>,
    ) -> Result<Option<TypeSet>, Error> {
        if let Some(name) = self.exact_name(node) {
            return Ok(Some(self.exact[&name].clone()));
        }
        match &node.kind {
            Kind::Name(name) => {
                if let Some(set) = self.exact.get(name).or_else(|| self.sets.get(name)) {
                    return Ok(Some(set.clone()));
                }
                if pending.contains_key(name) {
                    return Ok(None);
                }
                Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown type or typeset `{name}`"),
                ))
            }
            Kind::Call(name, args) if name == "vectors" && args.len() == 1 => {
                let Some(set) = self.member(source, &args[0], pending)? else {
                    return Ok(None);
                };
                if !set.is_empty() && !set.subset_of(&self.lanes) {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "vectors() requires a set of non-pointer scalar types",
                    ));
                }
                Ok(Some(set.vectors(self.max_exponent)))
            }
            Kind::Union(parts) | Kind::Intersection(parts) => {
                let mut result = None;
                let mut resolved = true;
                for part in parts {
                    match self.member(source, part, pending)? {
                        Some(set) => match &mut result {
                            None => result = Some(set),
                            Some(result) if matches!(node.kind, Kind::Union(_)) => {
                                result.union(&set)
                            }
                            Some(result) => result.intersect(&set),
                        },
                        None => resolved = false,
                    }
                }
                Ok(if resolved { result } else { None })
            }
            _ => Err(Error::at(
                source,
                node.offset,
                "expected a type, typeset or vectors(set)",
            )),
        }
    }

    pub fn set(&self, source: &str, node: &Node) -> Result<TypeSet, Error> {
        let set = self
            .member(source, node, &BTreeMap::new())?
            .expect("all named sets have been resolved");
        if set.is_empty() {
            return Err(Error::at(
                source,
                node.offset,
                "type constraint must not be empty",
            ));
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sets_preserve_width_lane_count_and_scalability() {
        let mut wide = TypeSet::singleton(Primitive::Int(32), 0, false);
        wide.union(&TypeSet::singleton(Primitive::Int(64), 0, false));
        let mut shapes = TypeSet::singleton(Primitive::Int(32), 2, false);
        shapes.union(&TypeSet::singleton(Primitive::Int(32), 2, true));
        let vectors = wide.vectors(15);
        for code in [
            Primitive::Int(8),
            Primitive::Int(16),
            Primitive::Int(32),
            Primitive::Int(64),
            Primitive::Float(32),
            Primitive::Bool,
            Primitive::Ptr,
        ] {
            for exponent in 0..=15 {
                for scalable in [false, true] {
                    let ty = TypeSet::singleton(code, exponent, scalable);
                    assert_eq!(
                        ty.subset_of(&wide),
                        matches!(code, Primitive::Int(32 | 64)) && exponent == 0 && !scalable
                    );
                    assert_eq!(
                        ty.subset_of(&shapes),
                        code == Primitive::Int(32) && exponent == 2
                    );
                    assert_eq!(
                        ty.subset_of(&vectors),
                        matches!(code, Primitive::Int(32 | 64)) && exponent > 0
                    );
                }
            }
        }
    }

    #[test]
    fn shape_constraints_retain_the_exact_type_set() {
        let mut integers = TypeSet::default();
        for bits in [8, 16, 32, 64] {
            integers.union(&TypeSet::singleton(Primitive::Int(bits), 0, false));
        }
        let mut set = integers.vectors(15);
        set.union(&integers);
        let fixed_four = TypeSet::singleton(Primitive::Int(32), 2, false);
        let fixed_two = TypeSet::singleton(Primitive::Int(64), 1, false);
        let four_lanes = 1 << 2;
        set.retain_shapes(four_lanes);
        assert_eq!(
            set.0,
            [8, 16, 32, 64]
                .map(|bits| (Primitive::Int(bits), four_lanes))
                .into()
        );
        set.intersect(&fixed_four);
        assert_eq!(set, fixed_four);
        set.intersect(&fixed_two);
        assert!(set.is_empty());
    }
}
