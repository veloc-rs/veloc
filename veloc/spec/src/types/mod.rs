//! Scalar types, named vectors and exact type-set expressions from types.ops.

use std::collections::BTreeMap;

use crate::Error;
use crate::model::Fields;
use crate::syntax::{Kind, Node, Record};

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

pub(crate) use veloc_types::Scalar as Primitive;

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
    pub predicates: BTreeMap<String, TypeSet>,
    pub lanes: TypeSet,
    pub integers: TypeSet,
    pub scalar_floats: TypeSet,
    max_exponent: u32,
}

pub(crate) fn rust_type(name: &str) -> String {
    if let Some(member) = name.strip_prefix("Type.") {
        format!("crate::Type::{member}")
    } else {
        format!("crate::types::{name}")
    }
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

    pub fn compile(records: &[Record], source: &str) -> Result<Self, Error> {
        if let Some(record) = records
            .iter()
            .find(|r| matches!(r.kind.as_str(), "type" | "typeset") && r.name == "Callable")
        {
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
            predicates: BTreeMap::new(),
            lanes: TypeSet::default(),
            integers: TypeSet::default(),
            scalar_floats: TypeSet::default(),
            max_exponent: veloc_types::MAX_VECTOR_LANES.trailing_zeros(),
        };
        for scalar in &types.scalars {
            let single = TypeSet::singleton(scalar.ty, 0, false);
            let mut family = single.clone();
            if scalar.ty != Primitive::Ptr {
                types.lanes.union(&single);
                family.union(&single.vectors(veloc_types::MAX_VECTOR_LANES.trailing_zeros()));
            }
            if matches!(scalar.ty, Primitive::Int(_) | Primitive::Bool) {
                types.integers.union(&family);
            }
            if matches!(scalar.ty, Primitive::Float(_)) {
                types.scalar_floats.union(&single);
            }
        }
        let mut pending = BTreeMap::new();
        for record in records.iter().filter(|r| r.kind == "typeset") {
            let mut fields = Fields::new(source, record.clone());
            if types.exact.contains_key(&record.name) {
                return Err(fields.error("typeset name shadows an exact type"));
            }
            if matches!(
                record.name.as_str(),
                "values" | "successor" | "successors" | "signature"
            ) {
                return Err(fields.error("typeset name shadows a signature keyword"));
            }
            let expr = fields.take("set")?;
            fields.finish()?;
            pending.insert(record.name.clone(), (record.offset, expr));
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
        for record in records.iter().filter(|r| r.kind == "predicate") {
            let mut fields = Fields::new(source, record.clone());
            // Predicate methods occupy the is_* namespace. These two methods
            // describe validity or physical shape, rather than a declared set.
            if !record.name.starts_with("is_")
                || record.name.len() == 3
                || !record
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            {
                return Err(fields.error("predicate name must be snake_case and start with is_"));
            }
            if matches!(
                record.name.as_str(),
                "is_valid"
                    | "is_scalable"
                    | "is_compact"
                    | "is_callable"
                    | "is_owned"
                    | "is_scalar"
                    | "is_vector"
                    | "is_integer"
                    | "is_float"
                    | "is_ptr"
                    | "is_predicate"
                    | "is_fixed"
                    | "is_local"
                    | "is_shared"
            ) {
                return Err(fields.error("predicate name conflicts with a built-in Type method"));
            }
            let set = types.set(source, &fields.take("set")?)?;
            fields.finish()?;
            types.predicates.insert(record.name.clone(), set);
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

    pub fn is_definition(kind: &str) -> bool {
        matches!(kind, "type" | "typeset" | "predicate")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sets_preserve_width_lane_count_and_scalability() {
        let defs = crate::fixtures::parse(
            r#"
            typeset Wide = Type.I32 | Type.I64;
            typeset Shapes = Type.I32X4 | SV4;
            type SV4 = vector(Type.I32, scalable(4));
            typeset AllWideVectors = vectors(Wide);
        "#,
        )
        .unwrap();
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
                        ty.subset_of(&defs.types.sets["Wide"]),
                        matches!(code, Primitive::Int(32 | 64)) && exponent == 0 && !scalable
                    );
                    assert_eq!(
                        ty.subset_of(&defs.types.sets["Shapes"]),
                        code == Primitive::Int(32) && exponent == 2
                    );
                    assert_eq!(
                        ty.subset_of(&defs.types.sets["AllWideVectors"]),
                        matches!(code, Primitive::Int(32 | 64)) && exponent > 0
                    );
                }
            }
        }
    }

    #[test]
    fn shape_constraints_retain_the_exact_type_set() {
        let types = crate::fixtures::types();
        let mut set = types.sets["Integer"].clone();
        set.retain_shapes(1 << 2); // Fixed vectors with four lanes.
        assert_eq!(
            set.0,
            BTreeMap::from([
                (Primitive::Int(8), 4),
                (Primitive::Int(16), 4),
                (Primitive::Int(32), 4),
                (Primitive::Int(64), 4)
            ])
        );
        set.intersect(&types.exact["Type.I32X4"]);
        assert_eq!(set, types.exact["Type.I32X4"]);
        set.intersect(&types.exact["Type.I64X2"]);
        assert!(set.is_empty());
    }
}
