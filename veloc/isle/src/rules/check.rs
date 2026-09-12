//! Universal source types are rigid; target types are inferred. A rule must
//! accept every source type, not merely have a nonempty intersection with it.
use super::{Call, Dialects, Expr, Rule};
use std::collections::BTreeMap;
use veloc_opgen::Error;
use veloc_opgen::schema::{Signature, Term, TypeSet};

#[derive(Clone, Copy, Debug)]
pub(super) enum TypeRef {
    Input(usize),
    Result(usize),
}

struct Ty {
    parent: usize,
    domain: TypeSet,
    anchor: Option<TypeRef>,
}

#[derive(Debug)]
pub(super) struct Inst {
    pub op: String,
    pub inputs: Vec<usize>,
    pub results: Vec<usize>,
}

#[derive(Debug)]
pub(super) struct CheckedRule {
    pub name: String,
    pub inputs: usize,
    pub types: Vec<TypeRef>,
    pub insts: Vec<Inst>,
    pub outputs: Vec<usize>,
}

struct Checker<'a> {
    source: &'a str,
    dialects: &'a Dialects,
    types: Vec<Ty>,
    values: Vec<usize>,
    bindings: BTreeMap<String, usize>,
    insts: Vec<Inst>,
}

impl Checker<'_> {
    fn fresh(&mut self, domain: &TypeSet, anchor: Option<TypeRef>) -> usize {
        let id = self.types.len();
        self.types.push(Ty {
            parent: id,
            domain: domain.clone(),
            anchor,
        });
        id
    }

    fn root(&self, mut ty: usize) -> usize {
        while self.types[ty].parent != ty {
            ty = self.types[ty].parent;
        }
        ty
    }

    fn unify(&mut self, a: usize, b: usize, offset: usize) -> Result<(), Error> {
        let a = self.root(a);
        let b = self.root(b);
        if a == b {
            return Ok(());
        }
        let left = &self.types[a];
        let right = &self.types[b];
        match (left.anchor, right.anchor) {
            (Some(_), Some(_)) => {
                if left.domain != right.domain || !left.domain.is_singleton() {
                    return Err(Error::at(
                        self.source,
                        offset,
                        "rule requires independent source types to be equal",
                    ));
                }
                self.types[b].parent = a;
            }
            (Some(_), None) => {
                if !left.domain.subset_of(&right.domain) {
                    return Err(Error::at(
                        self.source,
                        offset,
                        "target type domain does not cover every source type",
                    ));
                }
                self.types[b].parent = a;
            }
            (None, Some(_)) => return self.unify(b, a, offset),
            (None, None) => {
                let mut domain = left.domain.clone();
                domain.intersect(&right.domain);
                if domain.is_empty() {
                    return Err(Error::at(self.source, offset, "incompatible type domains"));
                }
                self.types[a].domain = domain;
                self.types[b].parent = a;
            }
        }
        Ok(())
    }

    fn signature(&mut self, signature: &Signature, source: bool) -> (Vec<usize>, Vec<usize>) {
        let mut variables = BTreeMap::new();
        let mut term = |this: &mut Self, term: &Term, anchor| {
            if let Some(id) = term.variable {
                if let Some(&ty) = variables.get(&id) {
                    return ty;
                }
            }
            let ty = this.fresh(&term.domain, if source { Some(anchor) } else { None });
            if let Some(id) = term.variable {
                variables.insert(id, ty);
            }
            ty
        };
        let inputs = signature
            .inputs
            .iter()
            .enumerate()
            .map(|(i, t)| term(self, t, TypeRef::Input(i)))
            .collect();
        let results = signature
            .results
            .iter()
            .enumerate()
            .map(|(i, t)| term(self, t, TypeRef::Result(i)))
            .collect();
        (inputs, results)
    }

    fn expression(&mut self, expr: Expr, expected: Option<&[usize]>) -> Result<Vec<usize>, Error> {
        match expr {
            Expr::Value(name, offset) => {
                let value = *self.bindings.get(&name).ok_or_else(|| {
                    Error::at(self.source, offset, format!("unbound value {name}"))
                })?;
                if let Some(expected) = expected {
                    if expected.len() != 1 {
                        return Err(Error::at(
                            self.source,
                            offset,
                            "value cannot supply multiple results",
                        ));
                    }
                    self.unify(self.values[value], expected[0], offset)?;
                }
                Ok(vec![value])
            }
            Expr::Call(call) => self.call(call, expected),
        }
    }

    fn call(&mut self, call: Call, expected: Option<&[usize]>) -> Result<Vec<usize>, Error> {
        let operation = self
            .dialects
            .operation(&call.op, self.source, call.offset)?;
        if operation.constrained {
            return Err(Error::at(
                self.source,
                call.offset,
                "target verifier preconditions require an explicit adapter",
            ));
        }
        let signature = operation
            .signature
            .as_ref()
            .map_err(|message| Error::at(self.source, call.offset, message))?;
        if call.args.len() != signature.inputs.len() {
            return Err(Error::at(
                self.source,
                call.offset,
                format!("wrong operand count for {}", call.op),
            ));
        }
        let (input_types, result_types) = self.signature(signature, false);
        if let Some(expected) = expected {
            if expected.len() != result_types.len() {
                return Err(Error::at(
                    self.source,
                    call.offset,
                    format!("wrong result count for {}", call.op),
                ));
            }
            for (&result, &expected) in result_types.iter().zip(expected) {
                self.unify(result, expected, call.offset)?;
            }
        }
        let mut inputs = Vec::new();
        for (arg, expected) in call.args.into_iter().zip(input_types) {
            inputs.extend(self.expression(arg, Some(&[expected]))?);
        }
        let results: Vec<_> = result_types
            .into_iter()
            .map(|ty| {
                let id = self.values.len();
                self.values.push(ty);
                id
            })
            .collect();
        self.insts.push(Inst {
            op: call.op,
            inputs,
            results: results.clone(),
        });
        Ok(results)
    }
}

pub(super) fn check(source: &str, dialects: &Dialects, rule: Rule) -> Result<CheckedRule, Error> {
    let operation = dialects.operation(&rule.root.op, source, rule.root.offset)?;
    let signature = operation
        .signature
        .as_ref()
        .map_err(|message| Error::at(source, rule.root.offset, message))?;
    if rule.root.args.len() != signature.inputs.len() {
        return Err(Error::at(
            source,
            rule.root.offset,
            "wrong source operand count",
        ));
    }
    let mut checker = Checker {
        source,
        dialects,
        types: Vec::new(),
        values: Vec::new(),
        bindings: BTreeMap::new(),
        insts: Vec::new(),
    };
    let (inputs, results) = checker.signature(signature, true);
    for (arg, ty) in rule.root.args.into_iter().zip(inputs) {
        let Expr::Value(name, offset) = arg else {
            return Err(Error::at(
                source,
                rule.root.offset,
                "nested source matching requires a graph adapter",
            ));
        };
        let id = checker.values.len();
        if checker.bindings.insert(name, id).is_some() {
            return Err(Error::at(
                source,
                offset,
                "repeated source binding requires an equality guard",
            ));
        }
        checker.values.push(ty);
    }
    let input_count = checker.values.len();
    let outputs = if rule.outputs.len() == 1 {
        checker.expression(rule.outputs.into_iter().next().unwrap(), Some(&results))?
    } else {
        if rule.outputs.len() != results.len() {
            return Err(Error::at(
                source,
                rule.root.offset,
                "wrong rule result count",
            ));
        }
        let mut outputs = Vec::new();
        for (expr, expected) in rule.outputs.into_iter().zip(results) {
            outputs.extend(checker.expression(expr, Some(&[expected]))?);
        }
        outputs
    };
    let types = checker
        .values
        .iter()
        .map(|&ty| {
            checker.types[checker.root(ty)].anchor.ok_or_else(|| {
                Error::at(
                    source,
                    rule.root.offset,
                    "temporary type cannot be inferred from source operands or results",
                )
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(CheckedRule {
        name: rule.name,
        inputs: input_count,
        types,
        insts: checker.insts,
        outputs,
    })
}
