//! Pure construction functions. Calls are type checked and inlined into one
//! construction plan, so parameter reuse never duplicates emitted instructions.
use super::typed::{Call, Inst, Signature, Ty, domain};
use crate::{
    Definitions, Error, interfaces,
    schema::Operation,
    syntax::{Decl, DeclKind, FunctionBody, Node, Results},
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

struct Function {
    signature: Signature,
    generics: Vec<String>,
    body: FunctionBody,
    offset: usize,
}

pub(super) struct Functions(BTreeMap<String, Function>);

fn same(a: &Ty, b: &Ty) -> bool {
    a == b || (a.domain.len() == 1 && a.domain == b.domain)
}

impl Functions {
    pub fn compile(
        source: &str,
        declarations: &[Decl],
        aliases: &BTreeMap<String, Node>,
    ) -> Result<Self, Error> {
        let mut functions = BTreeMap::new();
        for d in declarations {
            let DeclKind::Function { signature, body } = &d.kind else {
                continue;
            };
            if signature.is_const {
                return Err(Error::at(
                    source,
                    d.offset,
                    "construction functions build IR values, not const values",
                ));
            }
            if let FunctionBody::Rust { offset, path } = body {
                let path = path
                    .as_ref()
                    .ok_or_else(|| Error::at(source, *offset, "expected Rust function path"))?;
                interfaces::rust_path(source, *offset, path)?;
            }
            let mut sig = Signature {
                node: String::new(),
                dynamic: false,
                inputs: Vec::new(),
                results: Vec::new(),
                hosts: Vec::new(),
                generics: BTreeMap::new(),
            };
            let mut names = BTreeSet::new();
            for p in &signature.generics {
                if p.moves || !names.insert(p.name.clone()) {
                    return Err(Error::at(
                        source,
                        p.offset,
                        "duplicate or moved function generic",
                    ));
                }
                sig.generics.insert(
                    p.name.clone(),
                    Ty {
                        name: p.name.clone(),
                        domain: domain(source, &p.ty, aliases, &mut BTreeSet::new())?,
                    },
                );
            }
            for p in &signature.params {
                if p.moves || !names.insert(p.name.clone()) {
                    return Err(Error::at(
                        source,
                        p.offset,
                        "duplicate or moved function parameter",
                    ));
                }
                sig.inputs.push((p.name.clone(), sig.ty(source, &p.ty)?));
            }
            let Results::Fixed(results) = &signature.results else {
                return Err(Error::at(
                    source,
                    d.offset,
                    "function requires an explicit result type",
                ));
            };
            if results.len() != 1 {
                return Err(Error::at(
                    source,
                    d.offset,
                    "construction function requires one value result",
                ));
            }
            sig.results.push(sig.ty(source, &results[0].ty)?);
            let function = Function {
                signature: sig,
                generics: signature.generics.iter().map(|p| p.name.clone()).collect(),
                body: body.clone(),
                offset: d.offset,
            };
            if functions.insert(d.name.clone(), function).is_some() {
                return Err(Error::at(
                    source,
                    d.offset,
                    "duplicate construction function",
                ));
            }
        }
        Ok(Self(functions))
    }

    /// A generic wrapper checks every Rust binding, even when never selected.
    /// Its bound deliberately omits root lookup/replacement capabilities.
    pub fn wrappers(&self, ty: &str, bound: &str, value: &str) -> String {
        let mut out = String::new();
        for (name, f) in &self.0 {
            let FunctionBody::Rust {
                path: Some(path), ..
            } = &f.body
            else {
                continue;
            };
            let types = (0..f.generics.len())
                .map(|i| format!("ty{i}"))
                .collect::<Vec<_>>();
            let values = (0..f.signature.inputs.len())
                .map(|i| format!("arg{i}"))
                .collect::<Vec<_>>();
            let params = types
                .iter()
                .map(|n| format!("{n}: {ty}"))
                .chain(values.iter().map(|n| format!("{n}: {value}")))
                .collect::<Vec<_>>();
            let args = types.iter().chain(&values).cloned().collect::<Vec<_>>();
            writeln!(
                out,
                "fn build_{name}<C: {bound}>(ctx: &mut C, {}) -> {value} {{ {path}(ctx, {}) }}",
                params.join(", "),
                args.join(", ")
            )
            .unwrap();
        }
        out
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    // Check unused functions too, with abstract generic types rather than only
    // the concrete specializations reached by today's target rules.
    pub fn validate(
        &self,
        source: &str,
        operations: &BTreeMap<String, Operation>,
        defs: &Definitions,
        dialect: &str,
    ) -> Result<(), Error> {
        for (name, f) in &self.0 {
            for ty in f
                .signature
                .inputs
                .iter()
                .map(|(_, ty)| ty)
                .chain(&f.signature.results)
                .chain(f.signature.generics.values())
            {
                for member in &ty.domain {
                    if defs.type_domain(member).is_none() {
                        return Err(Error::at(
                            source,
                            f.offset,
                            format!("unknown OpSpec type {member}"),
                        ));
                    }
                }
            }
            let types = f
                .generics
                .iter()
                .map(|n| f.signature.generics[n].clone())
                .collect::<Vec<_>>();
            let args = f
                .signature
                .inputs
                .iter()
                .enumerate()
                .map(|(i, (_, ty))| (ty.clone(), format!("arg{i}")))
                .collect();
            self.call(
                source,
                f.offset,
                name,
                &types,
                args,
                operations,
                defs,
                &mut Vec::new(),
                dialect,
                &mut Vec::new(),
            )?;
        }
        Ok(())
    }

    pub fn call(
        &self,
        source: &str,
        offset: usize,
        name: &str,
        types: &[Ty],
        args: Vec<(Ty, String)>,
        operations: &BTreeMap<String, Operation>,
        defs: &Definitions,
        insts: &mut Vec<Inst>,
        dialect: &str,
        active: &mut Vec<String>,
    ) -> Result<(Ty, String), Error> {
        let f = &self.0[name];
        if active.iter().any(|n| n == name) {
            return Err(Error::at(
                source,
                offset,
                format!(
                    "recursive construction function: {} -> {name}",
                    active.join(" -> ")
                ),
            ));
        }
        if types.len() != f.generics.len() || args.len() != f.signature.inputs.len() {
            return Err(Error::at(
                source,
                offset,
                format!("wrong argument arity for function {name}"),
            ));
        }
        let mut bindings = BTreeMap::new();
        for (parameter, actual) in f.generics.iter().zip(types) {
            if !actual
                .domain
                .iter()
                .all(|ty| f.signature.generics[parameter].domain.contains(ty))
            {
                return Err(Error::at(
                    source,
                    offset,
                    format!("type argument outside domain of {name}::{parameter}"),
                ));
            }
            bindings.insert(parameter.clone(), actual.clone());
        }
        let resolve = |ty: &Ty| {
            bindings
                .get(&ty.name)
                .cloned()
                .unwrap_or_else(|| ty.clone())
        };
        for ((_, expected), (actual, _)) in f.signature.inputs.iter().zip(&args) {
            if !same(&resolve(expected), actual) {
                return Err(Error::at(
                    source,
                    offset,
                    format!("argument type mismatch in function {name}"),
                ));
            }
        }
        let expected = resolve(&f.signature.results[0]);
        if matches!(f.body, FunctionBody::Rust { .. }) {
            let result = format!("v{}", insts.len());
            insts.push(Inst {
                op: Call::Host {
                    name: name.into(),
                    types: types.to_vec(),
                },
                ty: expected.clone(),
                inputs: args.into_iter().map(|(_, v)| v).collect(),
                result: result.clone(),
            });
            return Ok((expected, result));
        }
        let FunctionBody::Value(body) = &f.body else {
            unreachable!()
        };
        let sig = Signature {
            node: String::new(),
            dynamic: false,
            inputs: Vec::new(),
            results: vec![expected.clone()],
            hosts: Vec::new(),
            generics: bindings,
        };
        let mut locals = f
            .signature
            .inputs
            .iter()
            .map(|(name, _)| name.clone())
            .zip(args)
            .collect();
        active.push(name.into());
        let result = sig.expression(
            source,
            body,
            operations,
            defs,
            insts,
            dialect,
            &mut locals,
            self,
            active,
        );
        active.pop();
        let result = result?;
        if !same(&result.0, &expected) {
            return Err(Error::at(
                source,
                f.offset,
                format!("result type mismatch in function {name}"),
            ));
        }
        Ok(result)
    }
}
