//! Finite logical signatures. Foreign constraints are deliberately not evaluated here.
use super::{Primitive, TypeSet, Types};
use crate::model::{Pattern, TypeDef};
use std::collections::BTreeMap;

pub(crate) struct Case {
    pub kinds: Vec<Primitive>,
    pub shapes: Vec<u32>,
    pub same: Vec<Option<usize>>,
}

enum Domain<'a> {
    Set(&'a TypeSet),
    Same(usize),
}

pub(crate) fn enumerate(
    types: &Types,
    signature: &TypeDef,
) -> Result<Option<Vec<Case>>, &'static str> {
    let (Some(inputs), Some(outputs)) =
        (signature.operands.patterns(), signature.results.patterns())
    else {
        return Ok(None);
    };
    let mut bindings = BTreeMap::new();
    let mut domains = Vec::new();
    for pattern in inputs.iter().chain(outputs) {
        let domain = match pattern {
            Pattern::Set(set) => Domain::Set(set),
            Pattern::Bind(var, set) => {
                bindings.insert(*var, domains.len());
                Domain::Set(set)
            }
            Pattern::Same(var) => Domain::Same(bindings[var]),
            Pattern::Exact(name) => Domain::Set(&types.exact[name]),
            _ => return Ok(None),
        };
        domains.push(domain);
    }
    let same = domains
        .iter()
        .map(|d| match d {
            Domain::Same(i) => Some(*i),
            _ => None,
        })
        .collect::<Vec<_>>();
    fn visit(
        domains: &[Domain<'_>],
        values: &mut Vec<(Primitive, u32)>,
        cases: &mut Vec<Case>,
        same: &[Option<usize>],
    ) -> Result<(), &'static str> {
        let Some(domain) = domains.get(values.len()) else {
            if cases.len() == 65_536 {
                return Err("signature has too many type combinations");
            }
            cases.push(Case {
                kinds: values.iter().map(|v| v.0).collect(),
                shapes: values.iter().map(|v| v.1).collect(),
                same: same.to_vec(),
            });
            return Ok(());
        };
        match domain {
            Domain::Same(index) => {
                values.push(values[*index]);
                visit(domains, values, cases, same)?;
                values.pop();
            }
            Domain::Set(set) => {
                for (&kind, &mask) in &set.0 {
                    values.push((kind, mask));
                    visit(domains, values, cases, same)?;
                    values.pop();
                }
            }
        }
        Ok(())
    }
    let mut cases = Vec::new();
    visit(&domains, &mut Vec::new(), &mut cases, &same)?;
    Ok(Some(cases))
}
