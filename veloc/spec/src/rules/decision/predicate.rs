//! Typed query operations compiled exclusively to matcher bytecode.
use super::*;
use crate::storage::operands::{Domain, Shape};
use crate::syntax::FunctionBody;

#[derive(Clone, PartialEq, Eq)]
pub(super) enum Test {
    Signature {
        results: Vec<Pattern>,
        inputs: Vec<Pattern>,
    },
    Type {
        value: OperandRef,
        ty: String,
    },
    Signed {
        field: usize,
        bits: usize,
        expected: bool,
    },
    Features(usize),
}

/// Type expressions stay symbolic until Rust evaluates the shared type codec.
#[derive(Clone, PartialEq, Eq)]
pub(super) enum Pattern {
    Exact(String),
    Set(Vec<String>),
    Bind(Vec<String>),
    Same(usize),
}

impl Expressions<'_> {
    pub(super) fn access(&self, node: &Node) -> Result<(Domain, usize), Error> {
        let fail = || {
            Error::at(
                self.source,
                node.offset,
                "expected a declared scalar instruction field",
            )
        };
        let Kind::Member(receiver, field) = &node.kind else {
            return Err(fail());
        };
        if !matches!(&receiver.kind, Kind::Name(n) if n == &self.signature.node) {
            return Err(fail());
        }
        let mut access = None;
        for root in self.roots {
            let op = self
                .defs
                .ops
                .iter()
                .find(|op| &op.name == root)
                .ok_or_else(fail)?;
            let crate::model::Projection::Operands(layout) = &op.projection else {
                return Err(fail());
            };
            let member = layout
                .members
                .iter()
                .find(|m| m.binding.as_deref() == Some(field))
                .ok_or_else(fail)?;
            if member.field.shape != Shape::One {
                return Err(fail());
            }
            if member.domain == Domain::Attribute && member.field.rust != "i64" {
                return Err(fail());
            }
            let current = (member.domain, member.index);
            if access.is_some_and(|old| old != current) {
                return Err(Error::at(
                    self.source,
                    node.offset,
                    "alternative operations have different field layouts",
                ));
            }
            access = Some(current);
        }
        access.ok_or_else(fail)
    }

    fn native<'n>(&self, node: &'n Node) -> Result<Option<(&str, &'n Node, &'n [Node])>, Error> {
        let Kind::Method(receiver, name, args) = &node.kind else {
            return Ok(None);
        };
        let owner = if let Kind::Name(host) = &receiver.kind {
            self.hosts
                .iter()
                .find(|(n, _)| n == host)
                .map(|(_, owner)| owner.as_str())
        } else if matches!(receiver.kind, Kind::Member(..)) {
            let (domain, _) = self.access(receiver)?;
            let rust = if domain == Domain::Attribute {
                "i64"
            } else {
                self.value_type
            };
            let mut owners = self.types.iter().filter(|(owner, declaration)| {
                self.bindings
                    .0
                    .get(**owner)
                    .is_some_and(|binding| binding.path == rust)
                    && declaration
                        .members()
                        .iter()
                        .any(|method| method.name == *name)
            });
            let owner = owners.next().map(|(owner, _)| *owner);
            if owners.next().is_some() {
                return Err(Error::at(
                    self.source,
                    node.offset,
                    "ambiguous field method declaration",
                ));
            }
            owner
        } else {
            None
        };
        let Some(owner) = owner else {
            return Ok(None);
        };
        let method = self.types[owner].members().iter().find(|d| d.name == *name);
        let Some(method) = method else {
            return Ok(None);
        };
        let Some(FunctionBody::Vm { opcode, .. }) = method.body() else {
            return Ok(None);
        };
        if method.signature().unwrap().params.len() != args.len() + 1 {
            return Err(Error::at(
                self.source,
                node.offset,
                "VM operation argument arity",
            ));
        }
        Ok(Some((opcode, receiver, args)))
    }

    pub(super) fn guard(
        &self,
        node: &Node,
        program: &mut Program,
        out: &mut Vec<Test>,
    ) -> Result<(), Error> {
        if let Kind::Binary("&&", lhs, rhs) = &node.kind {
            self.guard(lhs, program, out)?;
            return self.guard(rhs, program, out);
        }
        let (node, expected) = match &node.kind {
            Kind::Unary("!", inner) => (inner.as_ref(), false),
            _ => (node, true),
        };
        let fail = || {
            Error::at(
                self.source,
                node.offset,
                "unsupported native predicate expression",
            )
        };
        // Pure DSL predicate: no host object or Rust callback is involved.
        if let Kind::Call(name, args) = &node.kind
            && name == "fits_signed"
        {
            let [value, bits] = args.as_slice() else {
                return Err(Error::at(
                    self.source,
                    node.offset,
                    "fits_signed expects a value and a bit width",
                ));
            };
            let (domain, field) = self.access(value)?;
            if domain != Domain::Attribute {
                return Err(Error::at(
                    self.source,
                    value.offset,
                    "fits_signed expects an i64 instruction attribute",
                ));
            }
            let bits = match bits.kind {
                Kind::Integer(n) => n,
                Kind::Number(n) => n.into(),
                _ => {
                    return Err(Error::at(
                        self.source,
                        bits.offset,
                        "bit width must be an integer literal",
                    ));
                }
            };
            if !(1..=64).contains(&bits) {
                return Err(Error::at(
                    self.source,
                    node.offset,
                    "signed bit width must be in 1..=64",
                ));
            }
            out.push(Test::Signed {
                field,
                bits: bits as usize,
                expected,
            });
            return Ok(());
        }
        if let Kind::Binary("==", lhs, rhs) = &node.kind {
            if let Some(("type", value, [])) = self.native(lhs)? {
                if !expected {
                    return Err(fail());
                }
                let (domain, index) = self.access(value)?;
                if domain == Domain::Attribute {
                    return Err(fail());
                }
                let Kind::Name(ty) = &rhs.kind else {
                    return Err(fail());
                };
                if self.defs.type_domain(ty).is_none() {
                    return Err(fail());
                }
                out.push(Test::Type {
                    value: match domain {
                        Domain::Input => OperandRef::Input(index),
                        Domain::Result => OperandRef::Result(index),
                        Domain::Attribute => unreachable!("checked value field"),
                    },
                    ty: self.constant(ty, rhs.offset)?,
                });
                return Ok(());
            }
        }
        if let Some((opcode, receiver, args)) = self.native(node)? {
            match (opcode, args) {
                ("features", [features]) if expected => {
                    let Kind::Name(name) = &features.kind else {
                        return Err(fail());
                    };
                    let set = intern(&mut program.features, self.constant(name, features.offset)?);
                    let Kind::Name(host) = &receiver.kind else {
                        return Err(fail());
                    };
                    let words = format!("{host}.words()");
                    if program
                        .feature_source
                        .as_ref()
                        .is_some_and(|old| old != &words)
                    {
                        return Err(Error::at(
                            self.source,
                            node.offset,
                            "one feature context is required per program",
                        ));
                    }
                    program.feature_source = Some(words);
                    out.push(Test::Features(set));
                }
                _ => return Err(fail()),
            }
        } else {
            return Err(Error::at(
                self.source,
                node.offset,
                "condition has no VM implementation; Rust predicate callbacks are not supported",
            ));
        }
        Ok(())
    }
}

pub(super) fn validate(
    declarations: &[Decl],
    source: &str,
    bindings: &interfaces::Bindings,
    value: &str,
) -> Result<(), Error> {
    let known = bindings.0.keys().cloned().collect();
    for (owner, method) in crate::syntax::walk(declarations) {
        let Some(FunctionBody::Vm { opcode, .. }) = method.body() else {
            continue;
        };
        let Some(owner) = owner else {
            return Err(Error::at(
                source,
                method.offset,
                "VM bindings belong to query methods",
            ));
        };
        let sig = method.signature().unwrap();
        let expected: (Vec<String>, String) = match opcode.as_str() {
            "type" => (
                vec![],
                bindings
                    .0
                    .get("Type")
                    .ok_or_else(|| Error::at(source, method.offset, "missing Type binding"))?
                    .path
                    .clone(),
            ),
            "features" => (vec!["&[u64]".into()], "bool".into()),
            _ => {
                return Err(Error::at(
                    source,
                    method.offset,
                    format!("unknown VM operation {opcode}"),
                ));
            }
        };
        let params = sig
            .params
            .iter()
            .skip(1)
            .map(|p| interfaces::Type::parse(&p.ty, source, &known).map(|t| t.rust(bindings)))
            .collect::<Result<Vec<_>, _>>()?;
        let crate::syntax::Results::Fixed(results) = &sig.results else {
            return Err(Error::at(source, method.offset, "VM result signature"));
        };
        let result = results
            .first()
            .map(|r| interfaces::Type::parse(&r.ty, source, &known).map(|t| t.rust(bindings)))
            .transpose()?;
        let receiver = sig.params.first().is_some_and(|p| p.name == "self" && matches!(&p.ty.kind, Kind::Ref(n) if matches!(&n.kind, Kind::Name(n) if n == owner)));
        let owner_path = bindings.0.get(owner).map(|binding| binding.path.as_str());
        let valid_owner = match opcode.as_str() {
            "type" => owner_path == Some(value),
            _ => true,
        };
        if !receiver
            || !valid_owner
            || sig.is_const
            || !sig.generics.is_empty()
            || params != expected.0
            || results.len() != 1
            || result.as_ref() != Some(&expected.1)
        {
            return Err(Error::at(
                source,
                method.offset,
                format!("invalid signature for VM operation {opcode}"),
            ));
        }
        if opcode == "features" {
            let ty = declarations.iter().find(|d| d.name == owner).unwrap();
            let words = ty
                .members()
                .iter()
                .find(|m| m.name == "words")
                .ok_or_else(|| {
                    Error::at(
                        source,
                        method.offset,
                        "feature context requires declared words()",
                    )
                })?;
            let sig = words
                .signature()
                .ok_or_else(|| Error::at(source, words.offset, "words must be a method"))?;
            let crate::syntax::Results::Fixed(results) = &sig.results else {
                return Err(Error::at(source, words.offset, "words result signature"));
            };
            if sig.params.len() != 1
                || results.len() != 1
                || interfaces::Type::parse(&results[0].ty, source, &known)?.rust(bindings)
                    != "&[u64]"
                || !matches!(words.body(), Some(FunctionBody::Rust { .. }))
            {
                return Err(Error::at(
                    source,
                    words.offset,
                    "words must be a Rust method returning sequence(u64)",
                ));
            }
        }
    }
    Ok(())
}
