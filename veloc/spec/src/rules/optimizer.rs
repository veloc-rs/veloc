//! One frontend for optimizer rules; output selection does not change checking.
use super::{equivalence, expression, instruction};
use crate::{
    Definitions, Error, Source, SourceError,
    syntax::{DeclKind, Kind, Node},
    types::TypeSet,
};
use std::collections::{BTreeMap, BTreeSet};

/// One normalized rule, independent of its execution host.
pub(super) struct Case<'a> {
    pub offset: usize,
    pub operation: &'a str,
    pub args: &'a [Node],
    pub results: Vec<&'a [Node]>,
    pub generics: BTreeMap<String, TypeSet>,
    pub guard: Option<&'a Node>,
    pub replacement: &'a Node,
}

pub(crate) struct Output {
    pub folds: String,
    pub equivalences: String,
    pub rewrites: String,
}

pub(crate) fn generate(
    source: &Source,
    defs: &Definitions,
    dialect: &str,
    rust: &str,
) -> Result<Output, SourceError> {
    if !super::identifier(dialect) || !rust.split("::").all(super::identifier) {
        return Err(source.locate(Error::at(
            source.text(),
            0,
            "invalid optimizer Rust binding",
        )));
    }
    let cases = cases(source, defs).map_err(|e| source.locate(e))?;
    let (rules, instructions) =
        expression::compile(source, defs, dialect, &cases).map_err(|e| source.locate(e))?;
    let opcode = format!("{rust}::Opcode");
    let types = format!("{rust}::Type");
    Ok(Output {
        folds: expression::emit_attributes(defs)
            + &crate::model::control::generate(defs)
            + &expression::emit_folds(defs, &rules, &opcode, &types),
        equivalences: equivalence::generate(defs, &rules, &opcode)
            + &expression::emit_plans(defs, &rules, &opcode, &types),
        rewrites: instruction::generate(source, defs, dialect, rust, &instructions)?,
    })
}

pub(super) fn annotation(node: &Node) -> (&Node, Option<&[Node]>) {
    match &node.kind {
        Kind::Typed(value, types) => (value, Some(types)),
        _ => (node, None),
    }
}

/// Both hosts resolve the same logical operation and operand order. A capability
/// mismatch may select another host; an unknown operation or wrong arity cannot.
pub(super) fn operation<'a>(
    source: &str,
    defs: &'a Definitions,
    dialect: &str,
    path: &str,
    arity: usize,
    at: usize,
) -> Result<&'a crate::model::Op, Error> {
    let op = lookup(defs, dialect, path)
        .ok_or_else(|| Error::at(source, at, "unknown rule operation"))?;
    if op.params.len() != arity {
        return Err(Error::at(
            source,
            at,
            "rule operation operand arity mismatch",
        ));
    }
    Ok(op)
}

pub(super) fn lookup<'a>(
    defs: &'a Definitions,
    dialect: &str,
    path: &str,
) -> Option<&'a crate::model::Op> {
    let name = if path.contains("::") {
        path.strip_prefix(dialect)?.strip_prefix("::")?
    } else {
        path
    };
    defs.ops.iter().find(|op| op.name == name)
}

fn cases<'a>(source: &'a Source, defs: &Definitions) -> Result<Vec<Case<'a>>, Error> {
    let aliases: BTreeMap<_, _> = source
        .declarations()
        .iter()
        .filter_map(|decl| {
            if let DeclKind::TypeSet(value) = &decl.kind {
                Some((decl.name.clone(), value.clone()))
            } else {
                None
            }
        })
        .collect();
    let mut out = Vec::new();
    for decl in source.declarations() {
        let DeclKind::Optimizer { operation } = &decl.kind else {
            continue;
        };
        let fail = |at, message| Error::at(source.text(), at, message);
        let Some(Node {
            kind: Kind::List(cases),
            ..
        }) = decl.fields.get("cases")
        else {
            return Err(fail(decl.offset, "expected cases"));
        };
        if cases.is_empty() {
            return Err(fail(decl.offset, "expected nonempty cases"));
        }
        for case in cases {
            let Kind::Record(fields) = &case.kind else {
                unreachable!("parsed case");
            };
            let mut pattern = &fields["match"];
            let mut results = Vec::new();
            while let Kind::Typed(value, types) = &pattern.kind {
                results.push(types.as_slice());
                pattern = value;
            }
            let Kind::List(args) = &pattern.kind else {
                unreachable!("parsed case inputs");
            };
            let mut generics = BTreeMap::new();
            if let Some(Node {
                kind: Kind::Record(params),
                ..
            }) = fields.get("generics")
            {
                for (name, bound) in params {
                    let domain = defs.types.set(
                        source.text(),
                        &expand_domain(source.text(), bound, &aliases, &mut BTreeSet::new())?,
                    )?;
                    if domain.is_empty() {
                        return Err(fail(bound.offset, "empty type parameter domain"));
                    }
                    generics.insert(name.clone(), domain);
                }
            }
            out.push(Case {
                offset: case.offset,
                operation,
                args,
                results,
                generics,
                guard: fields.get("when"),
                replacement: &fields["emit"],
            });
        }
    }
    Ok(out)
}

fn expand_domain(
    source: &str,
    node: &Node,
    aliases: &BTreeMap<String, Node>,
    active: &mut BTreeSet<String>,
) -> Result<Node, Error> {
    if let Kind::Name(name) = &node.kind
        && let Some(alias) = aliases.get(name)
    {
        if !active.insert(name.clone()) {
            return Err(Error::at(source, node.offset, "cyclic type-set alias"));
        }
        let result = expand_domain(source, alias, aliases, active);
        active.remove(name);
        return result;
    }
    let mut result = node.clone();
    if let Kind::Union(parts) | Kind::Intersection(parts) | Kind::Call(_, parts) = &mut result.kind
    {
        for part in parts {
            *part = expand_domain(source, part, aliases, active)?;
        }
    }
    Ok(result)
}
