//! Typed rule signatures and value expression checking against OpSpec.
use crate::{
    Definitions, Error,
    schema::Operation,
    syntax::{Decl, DeclKind, Kind, Node, Results},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Ty {
    pub name: String,
    pub domain: Vec<String>,
}

pub(super) struct Signature {
    pub node: String,
    pub dynamic: bool,
    pub inputs: Vec<(String, Ty)>,
    pub results: Vec<Ty>,
    pub hosts: Vec<(String, String)>,
    pub generics: BTreeMap<String, Ty>,
}

fn named<'a>(source: &str, node: &'a Node) -> Result<&'a str, Error> {
    if let Kind::Name(name) = &node.kind {
        Ok(name)
    } else {
        Err(Error::at(source, node.offset, "expected a type name"))
    }
}

pub(super) fn domain(
    source: &str,
    node: &Node,
    aliases: &BTreeMap<String, Node>,
    active: &mut BTreeSet<String>,
) -> Result<Vec<String>, Error> {
    match &node.kind {
        Kind::Name(name) if name.contains("::") => Ok(vec![name.clone()]),
        Kind::Name(name) => {
            if !active.insert(name.clone()) {
                return Err(Error::at(source, node.offset, "cyclic type set"));
            }
            let alias = aliases.get(name).ok_or_else(|| {
                Error::at(source, node.offset, format!("unknown type set {name}"))
            })?;
            let result = domain(source, alias, aliases, active);
            active.remove(name);
            result
        }
        Kind::Union(nodes) => {
            let mut result = BTreeSet::new();
            for node in nodes {
                result.extend(domain(source, node, aliases, active)?);
            }
            Ok(result.into_iter().collect())
        }
        _ => Err(Error::at(
            source,
            node.offset,
            "expected a named type set or union",
        )),
    }
}

impl Signature {
    pub fn parse(
        source: &str,
        d: &Decl,
        aliases: &BTreeMap<String, Node>,
        operations: &BTreeMap<String, Operation>,
        defs: &Definitions,
        dialect: &str,
    ) -> Result<(Self, Vec<String>), Error> {
        let (DeclKind::Rule(sig) | DeclKind::Rewrite(sig)) = &d.kind else {
            return Err(Error::at(
                source,
                d.offset,
                "rules require an explicit node parameter",
            ));
        };
        if !matches!(&sig.results, Results::Fixed(r) if r.is_empty()) {
            return Err(Error::at(
                source,
                d.offset,
                "node rules do not declare value return types",
            ));
        }
        let mut result = Self {
            node: String::new(),
            inputs: Vec::new(),
            results: Vec::new(),
            hosts: Vec::new(),
            generics: BTreeMap::new(),
            dynamic: false,
        };
        let mut names = BTreeSet::new();
        for p in &sig.generics {
            if p.moves || !names.insert(&p.name) {
                return Err(Error::at(
                    source,
                    p.offset,
                    "duplicate or moved rule generic",
                ));
            }
            result.generics.insert(
                p.name.clone(),
                Ty {
                    name: p.name.clone(),
                    domain: domain(source, &p.ty, aliases, &mut BTreeSet::new())?,
                },
            );
        }
        let mut node = None;
        for p in &sig.params {
            if p.moves || !names.insert(&p.name) {
                return Err(Error::at(
                    source,
                    p.offset,
                    "duplicate or moved rule parameter",
                ));
            }
            if let Kind::Ref(owner) = &p.ty.kind {
                result
                    .hosts
                    .push((p.name.clone(), named(source, owner)?.into()));
            } else if node.replace(p).is_some() {
                return Err(Error::at(
                    source,
                    p.offset,
                    "one instruction node is required per decision rule",
                ));
            }
        }
        let node =
            node.ok_or_else(|| Error::at(source, d.offset, "missing instruction node parameter"))?;
        result.node = node.name.clone();
        let choices = match &node.ty.kind {
            Kind::Union(nodes) => nodes.as_slice(),
            _ => std::slice::from_ref(&node.ty),
        };
        let mut roots = Vec::new();
        for choice in choices {
            let (path, types) = match &choice.kind {
                Kind::Name(name) => (name, &[][..]),
                Kind::Call(name, types) => (name, types.as_slice()),
                _ => {
                    return Err(Error::at(
                        source,
                        choice.offset,
                        "expected a qualified instruction node type",
                    ));
                }
            };
            let name = path
                .strip_prefix(&format!("{dialect}::"))
                .ok_or_else(|| Error::at(source, choice.offset, "unknown instruction namespace"))?;
            let op = operations.get(name).ok_or_else(|| {
                Error::at(source, choice.offset, format!("unknown operation {name}"))
            })?;
            if types.len() != op.declaration.generics.len() {
                return Err(Error::at(
                    source,
                    choice.offset,
                    format!(
                        "{name} requires {} type arguments",
                        op.declaration.generics.len()
                    ),
                ));
            }
            let mut bindings = BTreeMap::new();
            for ((parameter, bound), actual) in op
                .declaration
                .generics
                .iter()
                .zip(&op.type_parameters)
                .zip(types)
            {
                let actual = result.ty(source, actual)?;
                for ty in &actual.domain {
                    if !defs.type_domain(ty).is_some_and(|set| set.subset_of(bound)) {
                        return Err(Error::at(
                            source,
                            choice.offset,
                            format!("{name} does not accept {ty}"),
                        ));
                    }
                }
                bindings.insert(parameter.name.as_str(), actual);
            }
            let value_type = |ty: &Node| -> Result<Option<Ty>, Error> {
                if let Kind::Call(name, args) = &ty.kind {
                    if name == "Value" && args.len() == 1 {
                        let name = named(source, &args[0])?;
                        return Ok(Some(match bindings.get(name) {
                            Some(ty) => ty.clone(),
                            None => result.ty(source, &args[0])?,
                        }));
                    }
                }
                Ok(None)
            };
            let mut inputs = Vec::new();
            let mut dynamic = false;
            for parameter in &op.declaration.params {
                if let Some(ty) = value_type(&parameter.ty)? {
                    inputs.push((format!("{}.{}", node.name, parameter.name), ty));
                } else if matches!(&parameter.ty.kind, Kind::Call(name, _) if name == "sequence") {
                    dynamic = true;
                }
            }
            let mut outputs = Vec::new();
            match &op.declaration.results {
                Results::Fixed(results) => {
                    for output in results {
                        outputs.push(value_type(&output.ty)?.ok_or_else(|| {
                            Error::at(source, choice.offset, "node result requires a value type")
                        })?);
                    }
                }
                Results::Signature => dynamic = true,
            }
            if roots.is_empty() {
                result.inputs = inputs;
                result.results = outputs;
                result.dynamic = dynamic;
            } else if result.inputs != inputs
                || result.results != outputs
                || result.dynamic != dynamic
            {
                return Err(Error::at(
                    source,
                    choice.offset,
                    "node alternatives must have identical named value signatures",
                ));
            }
            roots.push(name.to_owned());
        }
        for name in result.generics.keys() {
            if result.anchor(name).is_none() {
                return Err(Error::at(
                    source,
                    d.offset,
                    format!("unbound rule type {name}"),
                ));
            }
        }
        Ok((result, roots))
    }

    pub(super) fn ty(&self, source: &str, node: &Node) -> Result<Ty, Error> {
        let name = named(source, node)?;
        if let Some(ty) = self.generics.get(name) {
            return Ok(ty.clone());
        }
        if !name.contains("::") {
            return Err(Error::at(
                source,
                node.offset,
                format!("undeclared rule type {name}"),
            ));
        }
        Ok(Ty {
            name: name.into(),
            domain: vec![name.into()],
        })
    }

    pub fn anchor(&self, ty: &str) -> Option<(bool, usize)> {
        self.inputs
            .iter()
            .position(|(_, t)| t.name == ty)
            .map(|i| (false, i))
            .or_else(|| {
                self.results
                    .iter()
                    .position(|t| t.name == ty)
                    .map(|i| (true, i))
            })
    }

    pub fn check_call(
        &self,
        source: &str,
        offset: usize,
        op: &Operation,
        inputs: &[Ty],
        outputs: &[Ty],
        defs: &Definitions,
    ) -> Result<(), Error> {
        let sig = op
            .signature
            .as_ref()
            .or_else(|e| {
                op.integer_literal
                    .as_ref()
                    .or(op.attributed.as_ref())
                    .ok_or(e)
            })
            .map_err(|e| Error::at(source, offset, e))?;
        if inputs.len() != sig.inputs.len() || outputs.len() != sig.results.len() {
            return Err(Error::at(
                source,
                offset,
                format!("wrong value arity for {}", op.name),
            ));
        }
        let mut infer = crate::types::infer::Infer::<usize>::default();
        let (expected_inputs, expected_outputs) = infer.signature(sig, |_, _| None);
        let mut variables = BTreeMap::new();
        for (actual, expected) in inputs
            .iter()
            .chain(outputs)
            .zip(expected_inputs.into_iter().chain(expected_outputs))
        {
            let id = if let Some(&id) = variables.get(&actual.name) {
                id
            } else {
                let mut domain = crate::schema::TypeSet::default();
                for name in &actual.domain {
                    domain.union(defs.type_domain(name).ok_or_else(|| {
                        Error::at(source, offset, format!("unknown OpSpec type {name}"))
                    })?);
                }
                let id = infer.fresh(&domain, Some(variables.len()));
                variables.insert(actual.name.clone(), id);
                id
            };
            infer.unify(id, expected, offset, source).map_err(|e| Error::at(source, offset,
                format!("{} does not accept these types; requires equal operand/result types where declared: {}", op.name, e.message)))?;
        }
        Ok(())
    }

    pub fn expression(
        &self,
        source: &str,
        node: &Node,
        operations: &BTreeMap<String, Operation>,
        defs: &Definitions,
        insts: &mut Vec<Inst>,
        dialect: &str,
        locals: &mut BTreeMap<String, (Ty, String)>,
        functions: &super::functions::Functions,
        active: &mut Vec<String>,
    ) -> Result<(Ty, String), Error> {
        if let Kind::Name(name) = &node.kind {
            return locals
                .get(name)
                .cloned()
                .ok_or_else(|| Error::at(source, node.offset, format!("unbound value {name}")));
        }
        if let Kind::List(statements) = &node.kind {
            let Some((last, bindings)) = statements.split_last() else {
                return Err(Error::at(source, node.offset, "empty replacement"));
            };
            for binding in bindings {
                let Kind::Let(name, value) = &binding.kind else {
                    return Err(Error::at(
                        source,
                        binding.offset,
                        "expected let binding before replacement result",
                    ));
                };
                let value = self.expression(
                    source, value, operations, defs, insts, dialect, locals, functions, active,
                )?;
                if locals.insert(name.clone(), value).is_some() {
                    return Err(Error::at(
                        source,
                        binding.offset,
                        "duplicate replacement binding",
                    ));
                }
            }
            return self.expression(
                source, last, operations, defs, insts, dialect, locals, functions, active,
            );
        }
        let reference = match &node.kind {
            Kind::Member(receiver, member) => match &receiver.kind {
                Kind::Name(receiver) => Some(format!("{receiver}.{member}")),
                _ => None,
            },
            _ => None,
        };
        if let Some(name) = &reference {
            let (index, (_, ty)) = self
                .inputs
                .iter()
                .enumerate()
                .find(|(_, (n, _))| n == name)
                .ok_or_else(|| Error::at(source, node.offset, format!("unbound value {name}")))?;
            return Ok((ty.clone(), format!("input{index}")));
        }
        let call = match &node.kind {
            Kind::TypedCall(name, types, args) => Some((name, types.as_slice(), args.as_slice())),
            Kind::Call(name, args) => Some((name, &[][..], args.as_slice())),
            _ => None,
        };
        if let Some((name, types, args)) = call {
            if functions.contains(name) {
                let args = args
                    .iter()
                    .map(|arg| {
                        self.expression(
                            source, arg, operations, defs, insts, dialect, locals, functions,
                            active,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let types = types
                    .iter()
                    .map(|ty| self.ty(source, ty))
                    .collect::<Result<Vec<_>, _>>()?;
                return functions.call(
                    source,
                    node.offset,
                    name,
                    &types,
                    args,
                    operations,
                    defs,
                    insts,
                    dialect,
                    active,
                );
            }
        }
        let Kind::TypedCall(name, types, args) = &node.kind else {
            return Err(Error::at(
                source,
                node.offset,
                "replacement operations require explicit result types",
            ));
        };
        let [ty] = types.as_slice() else {
            return Err(Error::at(
                source,
                node.offset,
                "value expression requires one result type",
            ));
        };
        let ty = self.ty(source, ty)?;
        let op =
            operations
                .get(name.strip_prefix(&format!("{dialect}::")).ok_or_else(|| {
                    Error::at(source, node.offset, "unknown instruction namespace")
                })?)
                .ok_or_else(|| {
                    Error::at(
                        source,
                        node.offset,
                        format!("unknown replacement operation {name}"),
                    )
                })?;
        if op.constrained {
            return Err(Error::at(
                source,
                node.offset,
                "replacement verifier preconditions require a checked adapter",
            ));
        }
        if op.integer_literal.is_some() {
            let value = match args.as_slice() {
                [
                    Node {
                        kind: Kind::Number(n),
                        ..
                    },
                ] => i64::from(*n),
                [
                    Node {
                        kind: Kind::Integer(n),
                        ..
                    },
                ] => i64::try_from(*n)
                    .map_err(|_| Error::at(source, node.offset, "integer literal exceeds i64"))?,
                _ => return Err(Error::at(source, node.offset, "expected one i64 literal")),
            };
            self.check_call(
                source,
                node.offset,
                op,
                &[],
                std::slice::from_ref(&ty),
                defs,
            )?;
            let result = format!("v{}", insts.len());
            insts.push(Inst {
                op: Call::Integer(op.name.clone(), value),
                ty: ty.clone(),
                inputs: Vec::new(),
                result: result.clone(),
            });
            return Ok((ty, result));
        }
        let mut properties = Vec::new();
        if op.attributed.is_some() {
            if args.len() != op.declaration.params.len() {
                return Err(Error::at(
                    source,
                    node.offset,
                    "wrong attributed instruction arity",
                ));
            }
            for (index, ty, variant) in &op.attributes {
                let Kind::Name(name) = &args[*index].kind else {
                    return Err(Error::at(
                        source,
                        args[*index].offset,
                        "expected a declared attribute constant",
                    ));
                };
                if name.split_once("::").map(|(owner, _)| owner) != Some(ty.as_str()) {
                    return Err(Error::at(
                        source,
                        args[*index].offset,
                        "attribute constant type mismatch",
                    ));
                }
                properties.push((variant.clone(), name.clone()));
            }
        }
        let args = args
            .iter()
            .enumerate()
            .filter(|(index, _)| !op.attributes.iter().any(|(i, _, _)| i == index))
            .map(|(_, a)| {
                self.expression(
                    source, a, operations, defs, insts, dialect, locals, functions, active,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.check_call(
            source,
            node.offset,
            op,
            &args.iter().map(|(ty, _)| ty.clone()).collect::<Vec<_>>(),
            std::slice::from_ref(&ty),
            defs,
        )?;
        let result = format!("v{}", insts.len());
        insts.push(Inst {
            op: if properties.is_empty() {
                Call::Instruction(op.name.clone())
            } else {
                Call::Attributed(op.name.clone(), properties)
            },
            ty: ty.clone(),
            inputs: args.into_iter().map(|(_, value)| value).collect(),
            result: result.clone(),
        });
        Ok((ty, result))
    }
}

pub(super) struct Inst {
    pub op: Call,
    pub ty: Ty,
    pub inputs: Vec<String>,
    pub result: String,
}

pub(super) enum Call {
    Instruction(String),
    Integer(String, i64),
    Attributed(String, Vec<(String, String)>),
    Host { name: String, types: Vec<Ty> },
}
