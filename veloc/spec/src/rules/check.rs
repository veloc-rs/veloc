//! Universal source types are rigid; target types are inferred. A rule must
//! accept every source type, not merely have a nonempty intersection with it.
use super::{Call, Dialects, Expr, Rule};
use crate::Error;
use crate::types::infer::Infer;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub(super) enum TypeRef {
    Input(usize),
    Result(usize),
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
    types: Infer<TypeRef>,
    values: Vec<usize>,
    bindings: BTreeMap<String, usize>,
    insts: Vec<Inst>,
}

impl Checker<'_> {
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
                    self.types
                        .unify(self.values[value], expected[0], offset, self.source)?;
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
        let (input_types, result_types) = self.types.signature(signature, |_, _| None);
        if let Some(expected) = expected {
            if expected.len() != result_types.len() {
                return Err(Error::at(
                    self.source,
                    call.offset,
                    format!("wrong result count for {}", call.op),
                ));
            }
            for (&result, &expected) in result_types.iter().zip(expected) {
                self.types
                    .unify(result, expected, call.offset, self.source)?;
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
        types: Infer::default(),
        values: Vec::new(),
        bindings: BTreeMap::new(),
        insts: Vec::new(),
    };
    let (inputs, results) = checker.types.signature(signature, |result, i| {
        Some(if result {
            TypeRef::Result(i)
        } else {
            TypeRef::Input(i)
        })
    });
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
            checker.types.anchor(ty).ok_or_else(|| {
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
