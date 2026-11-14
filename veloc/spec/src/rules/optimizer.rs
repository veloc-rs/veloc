//! One frontend for optimizer rules; output selection does not change checking.
use super::{equivalence, expression, instruction};
use crate::{
    Definitions, Error, Source, SourceError,
    syntax::{DeclKind, Kind, Node, Results},
};
use std::collections::BTreeMap;

/// A normalized case. An explicit result type describes a value equality;
/// otherwise the replacement preserves the matched instruction's SSA results.
pub(super) struct Case<'a> {
    pub offset: usize,
    pub root: &'a str,
    pub operation: &'a str,
    pub args: &'a [Node],
    pub result: Option<&'a Node>,
    pub generics: Option<&'a BTreeMap<String, Node>>,
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
    let cases = cases(source).map_err(|e| source.locate(e))?;
    let rules = expression::compile(source, defs, dialect, &cases).map_err(|e| source.locate(e))?;
    let opcode = format!("{rust}::Opcode");
    let types = format!("{rust}::Type");
    Ok(Output {
        folds: expression::emit_attributes(defs)
            + &crate::model::control::generate(defs)
            + &expression::emit_folds(defs, &rules, &opcode, &types),
        equivalences: equivalence::generate(defs, &rules, &opcode)
            + &expression::emit_plans(defs, &rules, &opcode, &types),
        rewrites: instruction::generate(source, defs, dialect, rust, &cases)?,
    })
}

fn cases(source: &Source) -> Result<Vec<Case<'_>>, Error> {
    let mut out = Vec::new();
    for decl in source.declarations() {
        let DeclKind::Rule(sig) = &decl.kind else {
            continue;
        };
        let fail = |at, message| Error::at(source.text(), at, message);
        let [root] = sig.params.as_slice() else {
            return Err(fail(decl.offset, "optimizer rule requires one root"));
        };
        if root.moves
            || !sig.generics.is_empty()
            || !matches!(&sig.results, Results::Fixed(r) if r.is_empty())
            || decl.fields.keys().any(|k| k != "cases")
        {
            return Err(fail(
                decl.offset,
                "optimizer groups declare only a root and cases",
            ));
        }
        let Kind::Name(operation) = &root.ty.kind else {
            return Err(fail(
                root.ty.offset,
                "declare the root operation without type arguments",
            ));
        };
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
                return Err(fail(case.offset, "expected case"));
            };
            if fields
                .keys()
                .any(|k| !matches!(k.as_str(), "match" | "emit" | "when" | "generics"))
            {
                return Err(fail(case.offset, "unknown optimizer case field"));
            }
            let pattern = fields
                .get("match")
                .ok_or_else(|| fail(case.offset, "expected root pattern"))?;
            let (args, result) = match &pattern.kind {
                Kind::Call(name, args) if name == &root.name => (args.as_slice(), None),
                Kind::TypedCall(name, types, args) if name == &root.name && types.len() == 1 => {
                    (args.as_slice(), Some(&types[0]))
                }
                _ => {
                    return Err(fail(
                        pattern.offset,
                        "expected the declared root: root(...) or root<T>(...)",
                    ));
                }
            };
            let generics = match fields.get("generics") {
                Some(Node {
                    kind: Kind::Record(params),
                    ..
                }) if result.is_some() => Some(params),
                Some(params) => {
                    return Err(fail(
                        params.offset,
                        "type parameters require a value pattern: root<T>(...)",
                    ));
                }
                None => None,
            };
            let replacement = fields
                .get("emit")
                .ok_or_else(|| fail(case.offset, "expected replacement"))?;
            out.push(Case {
                offset: case.offset,
                root: &root.name,
                operation,
                args,
                result,
                generics,
                guard: fields.get("when"),
                replacement,
            });
        }
    }
    Ok(out)
}
