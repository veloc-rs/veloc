//! Scalar types, named vectors and exact type-set expressions from types.ops.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::encoding::TypeEncoding;
use crate::model::{Fields, list};
use crate::syntax::{Kind, Node, Record};
use crate::type_set::TypeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ScalarKind {
    Integer,
    Float,
    Boolean,
    Pointer,
}

#[derive(Debug)]
pub(crate) struct Scalar {
    pub name: String,
    pub code: u8,
    pub bits: Option<u32>,
    pub kind: ScalarKind,
}

impl Scalar {
    pub fn exact(&self) -> String {
        self.name.to_ascii_uppercase()
    }
}

pub(crate) struct Vector {
    pub name: String,
    pub element: usize,
    pub lanes: u16,
    pub scalable: bool,
}

pub(crate) struct Types {
    pub scalars: Vec<Scalar>,
    pub vectors: Vec<Vector>,
    pub aliases: Vec<(String, usize)>,
    pub exact: BTreeMap<String, TypeSet>,
    pub classes: BTreeMap<String, TypeSet>,
    pub predicates: BTreeMap<String, TypeSet>,
    pub lanes: TypeSet,
    pub integers: TypeSet,
    pub scalar_floats: TypeSet,
    max_exponent: u32,
}

impl Types {
    pub fn compile(
        records: &[Record],
        source: &str,
        encoding: &TypeEncoding,
    ) -> Result<Self, Error> {
        let declarations = crate::type_expr::compile(records, source, encoding)?;
        let mut types = Self {
            scalars: declarations.scalars,
            vectors: declarations.vectors,
            aliases: declarations.aliases,
            exact: declarations.exact,
            classes: BTreeMap::new(),
            predicates: BTreeMap::new(),
            lanes: TypeSet::default(),
            integers: TypeSet::default(),
            scalar_floats: TypeSet::default(),
            max_exponent: encoding.lanes_log2_max(),
        };
        for scalar in &types.scalars {
            let single = TypeSet::singleton(scalar.code, 0, false);
            let mut family = single.clone();
            if scalar.kind != ScalarKind::Pointer {
                types.lanes.union(&single);
                family.union(&single.vectors(encoding.lanes_log2_max()));
            }
            if matches!(scalar.kind, ScalarKind::Integer | ScalarKind::Boolean) {
                types.integers.union(&family);
            }
            if scalar.kind == ScalarKind::Float {
                types.scalar_floats.union(&single);
            }
        }
        let mut classes = BTreeMap::new();
        for record in records.iter().filter(|r| r.kind == "class") {
            let mut fields = Fields::new(source, record.clone());
            if types.exact.contains_key(&record.name) {
                return Err(fields.error("class name shadows an exact type"));
            }
            if matches!(
                record.name.as_str(),
                "values" | "successor" | "successors" | "signature"
            ) {
                return Err(fields.error("class name shadows a signature keyword"));
            }
            let members = list(source, fields.take("members")?)?;
            if members.is_empty() {
                return Err(fields.error("type class must not be empty"));
            }
            fields.finish()?;
            classes.insert(record.name.clone(), (record.offset, members));
        }
        while !classes.is_empty() {
            let before = classes.len();
            for key in classes.keys().cloned().collect::<Vec<_>>() {
                let (_, members) = &classes[&key];
                let mut set = TypeSet::default();
                let mut resolved = true;
                let mut seen = BTreeSet::new();
                for member in members {
                    // Offsets are intentionally excluded from duplicate detection.
                    if !seen.insert(member_key(source, member)?) {
                        return Err(Error::at(source, member.offset, "duplicate class member"));
                    }
                    match types.member(source, member, &classes)? {
                        Some(value) => set.union(&value),
                        None => resolved = false,
                    }
                }
                if resolved {
                    if set.is_empty() {
                        return Err(Error::at(
                            source,
                            classes[&key].0,
                            "type class must not be empty",
                        ));
                    }
                    classes.remove(&key);
                    types.classes.insert(key, set);
                }
            }
            if classes.len() == before {
                let (name, (offset, _)) = classes.first_key_value().unwrap();
                return Err(Error::at(
                    source,
                    *offset,
                    format!("cyclic type class `{name}`"),
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
            if matches!(record.name.as_str(), "is_valid" | "is_scalable") {
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
        pending: &BTreeMap<String, (usize, Vec<Node>)>,
    ) -> Result<Option<TypeSet>, Error> {
        match &node.kind {
            Kind::Name(name) => {
                if let Some(set) = self.exact.get(name).or_else(|| self.classes.get(name)) {
                    return Ok(Some(set.clone()));
                }
                if pending.contains_key(name) {
                    return Ok(None);
                }
                Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown type or class `{name}`"),
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
                "expected a type, class or vectors(set)",
            )),
        }
    }

    pub fn set(&self, source: &str, node: &Node) -> Result<TypeSet, Error> {
        let set = self
            .member(source, node, &BTreeMap::new())?
            .expect("all named classes have been resolved");
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
        matches!(kind, "type" | "class" | "predicate")
    }
}

fn member_key(source: &str, node: &Node) -> Result<String, Error> {
    match &node.kind {
        Kind::Name(name) => Ok(name.clone()),
        Kind::Call(name, args) if name == "vectors" && args.len() == 1 => {
            Ok(format!("vectors({})", member_key(source, &args[0])?))
        }
        Kind::Union(parts) | Kind::Intersection(parts) => {
            let separator = if matches!(node.kind, Kind::Union(_)) {
                " | "
            } else {
                " & "
            };
            let parts = parts
                .iter()
                .map(|part| member_key(source, part))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("({})", parts.join(separator)))
        }
        _ => Err(Error::at(
            source,
            node.offset,
            "expected a type, class or vectors(set)",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sets_preserve_width_lane_count_and_scalability() {
        let defs = crate::fixtures::parse(
            r#"
            class Wide { members: [I32, I64] }
            class Shapes { members: [I32X4, SV4] }
            type SV4 = vector(I32, scalable(4));
            class AllWideVectors { members: [vectors(Wide)] }
        "#,
        )
        .unwrap();
        for code in 0..=u8::MAX {
            for exponent in 0..=15 {
                for scalable in [false, true] {
                    let ty = TypeSet::singleton(code, exponent, scalable);
                    assert_eq!(
                        ty.subset_of(&defs.types.classes["Wide"]),
                        matches!(code, 3 | 4) && exponent == 0 && !scalable
                    );
                    assert_eq!(
                        ty.subset_of(&defs.types.classes["Shapes"]),
                        code == 3 && exponent == 2
                    );
                    assert_eq!(
                        ty.subset_of(&defs.types.classes["AllWideVectors"]),
                        matches!(code, 3 | 4) && exponent > 0
                    );
                }
            }
        }
    }

    #[test]
    fn shape_constraints_retain_the_exact_type_set() {
        let types = crate::fixtures::types();
        let mut set = types.classes["Integer"].clone();
        set.retain_shapes(types.exact["I32X4"].shapes());
        assert_eq!(set.0, BTreeMap::from([(1, 4), (2, 4), (3, 4), (4, 4)]));
        set.intersect(&types.exact["I32X4"]);
        assert_eq!(set, types.exact["I32X4"]);
        set.intersect(&types.exact["I64X2"]);
        assert!(set.is_empty());
    }
}
