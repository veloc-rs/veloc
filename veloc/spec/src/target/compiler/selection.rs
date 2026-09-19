//! Logical selection interfaces come from operation signatures. Storage is only
//! a projection used after checking and simplifying the logical field accesses.
use crate::schema::{Contract, Operand, ValueType};
use crate::syntax::Kind;
use crate::target::{Pattern, PatternArg, SelectRuleDef};
use crate::types::Types;
use std::collections::BTreeMap;

struct Field<'a> {
    value: bool,
    ty: Option<&'a ValueType>,
    access: Option<String>,
}

fn path(owner: &str, field: &str) -> String {
    if owner.is_empty() {
        field.into()
    } else {
        format!("{owner}.{field}")
    }
}

fn variable(field: &str) -> String {
    format!("__field_{}", field.replace('.', "_"))
}

fn slot(ty: &ValueType) -> Option<u8> {
    match ty {
        ValueType::Bind(id, _) | ValueType::Same(id) => Some(*id),
        _ => None,
    }
}

/// The caller promises valid input IR. Equal-type constraints can therefore
/// share one guard; this is not an instruction verifier.
pub(super) fn resolve(
    rule: &mut SelectRuleDef,
    dialect: &str,
    contracts: &[Contract],
    types: &Types,
) -> Result<(), String> {
    let mut fields = BTreeMap::new();
    let mut requested = Vec::new();
    for (owner, opcode, schema, type_args) in
        std::iter::once(("", &mut rule.opcode, &mut rule.schema, &rule.type_args)).chain(
            rule.definitions
                .iter_mut()
                .map(|p| (p.name.as_str(), &mut p.opcode, &mut p.schema, &p.type_args)),
        )
    {
        let (namespace, name) = opcode
            .split_once("::")
            .ok_or("expected a qualified operation")?;
        if namespace != dialect {
            return Err(format!("unknown selection dialect {namespace}"));
        }
        let contract = contracts
            .iter()
            .find(|op| op.name == name)
            .ok_or_else(|| format!("unknown input operation {opcode}"))?;
        let Some(crate::syntax::Node {
            kind: Kind::Object(storage, mapping),
            ..
        }) = contract.fields.get("storage")
        else {
            return Err(format!("{opcode} requires a named storage projection"));
        };
        let mut logical = BTreeMap::new();
        let mut inputs = contract
            .signature
            .operands
            .patterns()
            .unwrap_or_default()
            .iter();
        for input in &contract.inputs {
            let (name, value, ty) = match input {
                Operand::Value(name) => (name, true, inputs.next()),
                Operand::Attribute { name, .. }
                | Operand::Values(name)
                | Operand::Successor(name)
                | Operand::Successors(name) => (name, false, None),
            };
            logical.insert(name.as_str(), (value, ty));
        }
        for (name, ty) in contract
            .results
            .iter()
            .zip(contract.signature.results.patterns().unwrap_or_default())
        {
            logical.insert(name.as_str(), (true, Some(ty)));
        }
        // Invert direct projections. Hidden/default storage members do not become
        // logical fields; non-invertible projections get a diagnostic when used.
        for (name, (value, ty)) in &logical {
            let access = mapping.iter().find_map(|(physical, value)| {
                matches!(&value.kind, Kind::Name(binding) if binding == name)
                    .then(|| path(owner, physical))
            });
            fields.insert(
                path(owner, name),
                Field {
                    value: *value,
                    ty: *ty,
                    access,
                },
            );
        }
        if !type_args.is_empty() {
            if type_args.len() != contract.generics.len() {
                return Err(format!(
                    "{opcode} requires {} type arguments",
                    contract.generics.len()
                ));
            }
            for (id, domain) in type_args.iter().enumerate() {
                let (anchor, bound) = logical
                    .iter()
                    .find_map(|(name, (_, ty))| match ty {
                        Some(ValueType::Bind(index, bound)) if usize::from(*index) == id => {
                            Some((*name, bound))
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        format!(
                            "no type anchor for {opcode} generic {}",
                            contract.generics[id]
                        )
                    })?;
                for ty in domain {
                    let member = ty.rsplit("::").next().unwrap();
                    if !types
                        .exact
                        .get(member)
                        .is_some_and(|actual| actual.subset_of(bound))
                    {
                        return Err(format!(
                            "{opcode} generic {} does not accept {member}",
                            contract.generics[id]
                        ));
                    }
                }
                requested.push((path(owner, anchor), domain.clone()));
            }
        }
        *schema = storage.clone();
        *opcode = name.into();
    }
    // Typed operation references add a guard to the generic's defining value.
    for (field, types) in requested {
        let test = Pattern::Typed {
            name: variable(&field),
            types,
        };
        if let Some(PatternArg::Named { pattern, .. }) = rule
            .fields
            .iter_mut()
            .find(|arg| matches!(arg, PatternArg::Named { name, .. } if name == &field))
        {
            *pattern = Box::new(Pattern::And(vec![(**pattern).clone(), test]));
        } else {
            rule.fields.push(PatternArg::Named {
                name: field,
                pattern: Box::new(test),
            });
        }
    }
    for producer in &rule.definitions {
        if !fields.get(&producer.input).is_some_and(|field| field.value) {
            return Err(format!(
                "def input {} must be a logical SSA value",
                producer.input
            ));
        }
    }
    let mut values = fields
        .iter()
        .filter(|(_, field)| field.value)
        .map(|(name, _)| variable(name))
        .collect::<std::collections::BTreeSet<_>>();
    for (temp, exemplar) in &rule.temps {
        if !values.contains(exemplar) {
            return Err(format!("temporary {temp} requires a logical SSA exemplar"));
        }
        values.insert(temp.clone());
    }
    let mut groups = BTreeMap::<(String, u8), Vec<(usize, Vec<String>)>>::new();
    for (index, arg) in rule.fields.iter().enumerate() {
        let PatternArg::Named { name, pattern } = arg else {
            unreachable!()
        };
        let field = fields
            .get(name)
            .ok_or_else(|| format!("unknown logical field {name}"))?;
        let mut constraints = Vec::new();
        domains(pattern, &mut constraints);
        if !constraints.is_empty() && !field.value {
            return Err(format!("type_is requires a logical SSA value, not {name}"));
        }
        if let Some(id) = field.ty.and_then(slot) {
            let owner = name.split_once('.').map_or("", |(owner, _)| owner);
            for domain in constraints {
                groups
                    .entry((owner.into(), id))
                    .or_default()
                    .push((index, domain));
            }
        }
    }
    for uses in groups.values() {
        let mut intersection = uses[0].1.clone();
        for (_, domain) in &uses[1..] {
            intersection.retain(|ty| domain.contains(ty));
        }
        // Keep contradictory explicit predicates as written. They simply cannot
        // match valid input; never weaken them by dropping one side.
        if uses.len() < 2 || intersection.is_empty() {
            continue;
        }
        let first = uses[0].0;
        for (index, _) in uses {
            let PatternArg::Named { pattern, .. } = &mut rule.fields[*index] else {
                unreachable!()
            };
            clear_domains(pattern);
        }
        let PatternArg::Named { name, pattern } = &mut rule.fields[first] else {
            unreachable!()
        };
        *pattern = Box::new(Pattern::And(vec![
            (**pattern).clone(),
            Pattern::Typed {
                name: variable(name),
                types: intersection,
            },
        ]));
    }
    // Only now lower logical names to the selected storage layout.
    for arg in &mut rule.fields {
        let PatternArg::Named { name, .. } = arg else {
            unreachable!()
        };
        *name = fields[name]
            .access
            .clone()
            .ok_or_else(|| format!("logical field {name} has no direct storage projection"))?;
    }
    for producer in &mut rule.definitions {
        producer.input = fields[&producer.input].access.clone().ok_or_else(|| {
            format!(
                "def input {} has no direct storage projection",
                producer.input
            )
        })?;
    }
    Ok(())
}

fn domains(pattern: &Pattern, out: &mut Vec<Vec<String>>) {
    match pattern {
        Pattern::Typed { types, .. } => out.push(types.clone()),
        Pattern::And(parts) => {
            for part in parts {
                domains(part, out);
            }
        }
        _ => {}
    }
}

fn clear_domains(pattern: &mut Pattern) {
    match pattern {
        Pattern::Typed { name, .. } => *pattern = Pattern::Variable(name.clone()),
        Pattern::And(parts) => {
            for part in parts {
                clear_domains(part);
            }
        }
        _ => {}
    }
}
