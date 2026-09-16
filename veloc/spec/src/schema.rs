//! Read-only logical contracts for rule compilers. No runtime IR or storage ABI.
use std::collections::BTreeMap;

use crate::Definitions;
use crate::model::{ParamKind, Pattern};
pub use crate::model::{Pattern as ValueType, TypeDef as ValueSignature, TypeList as ValueTypes};
pub use crate::types::TypeSet;

/// A checked logical operand. Its role never depends on an IR's physical storage.
#[derive(Debug)]
pub enum Operand {
    Value(String),
    Attribute { name: String, ty: String },
    Values(String),
    Successor(String),
    Successors(String),
}

/// Storage-independent instruction contract, consumed by target descriptions.
/// Consumer-specific fields must be checked and consumed by that consumer.
#[derive(Debug)]
pub struct Contract {
    pub name: String,
    pub offset: usize,
    pub inputs: Vec<Operand>,
    pub results: Vec<String>,
    pub signature: ValueSignature,
    pub fields: BTreeMap<String, crate::syntax::Node>,
}

impl crate::Source {
    /// Generate definition-owned data types without requiring an IR storage ABI.
    pub fn data_types(&self) -> Result<String, crate::SourceError> {
        let compile = || {
            let data = crate::model::data::Types::compile(self.declarations(), self.text())?;
            crate::source::scopes::check(self.text(), self.declarations(), self.files(), &data)?;
            Ok(data.generate(&[], None))
        };
        compile().map_err(|e| self.locate(e))
    }

    /// Shared, typed expressions for a consumer's generated Rust projections.
    pub fn expressions(&self) -> Result<Expressions<'_>, crate::SourceError> {
        let compile = || {
            Ok(Expressions {
                source: self,
                types: crate::types::Types::compile(self.declarations(), self.text())?,
                encodings: crate::model::encoding::compile(self.declarations(), self.text())?,
                data: crate::model::data::Types::compile(self.declarations(), self.text())?,
                library: crate::model::expr::Library::default(),
            })
        };
        compile().map_err(|e| self.locate(e))
    }

    /// Check instruction signatures without requiring a runtime storage strategy.
    pub fn contracts(&self) -> Result<Vec<Contract>, crate::SourceError> {
        let compile = || {
            use crate::model::{self, ParamKind};
            use crate::syntax::{DeclKind, Results};
            let records = self.declarations();
            let source = self.text();
            let types = crate::types::Types::compile(records, source)?;
            let encodings = model::encoding::compile(records, source)?;
            let data = model::data::Types::compile(records, source)?;
            let vocabulary = model::Vocabulary {
                types: &types,
                encodings: &encodings,
                data: &data,
            };
            let mut names = std::collections::BTreeSet::new();
            let mut contracts = Vec::new();
            for record in records {
                let DeclKind::Op(signature) = &record.kind else {
                    continue;
                };
                model::identifier(source, record.offset, &record.name)?;
                if !names.insert(&record.name) {
                    return Err(crate::Error::at(
                        source,
                        record.offset,
                        format!("duplicate op `{}`", record.name),
                    ));
                }
                let checked = model::operation::signature(
                    source,
                    record.offset,
                    signature.clone(),
                    vocabulary,
                )?;
                let results = match &signature.results {
                    Results::Fixed(results) => results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            r.name.clone().unwrap_or_else(|| {
                                if results.len() == 1 {
                                    "result".into()
                                } else {
                                    format!("result{i}")
                                }
                            })
                        })
                        .collect(),
                    Results::Signature => Vec::new(),
                };
                let inputs = checked
                    .params
                    .into_iter()
                    .map(|param| match param.kind {
                        ParamKind::Value => Operand::Value(param.name),
                        ParamKind::Property(ty) => Operand::Attribute {
                            name: param.name,
                            ty,
                        },
                        ParamKind::Values => Operand::Values(param.name),
                        ParamKind::Successor => Operand::Successor(param.name),
                        ParamKind::Successors => Operand::Successors(param.name),
                    })
                    .collect();
                contracts.push(Contract {
                    name: record.name.clone(),
                    offset: record.offset,
                    inputs,
                    results,
                    signature: checked.types,
                    fields: record.fields.clone(),
                });
            }
            crate::source::scopes::check(source, records, self.files(), &data)?;
            Ok(contracts)
        };
        compile().map_err(|e| self.locate(e))
    }
}

/// Target consumers supply typed operand projections, not an expression parser.
pub struct Expressions<'a> {
    source: &'a crate::Source,
    types: crate::types::Types,
    encodings: crate::model::encoding::Encodings,
    data: crate::model::data::Types,
    library: crate::model::expr::Library,
}
impl Expressions<'_> {
    pub fn rust(
        &mut self,
        node: &crate::syntax::Node,
        result: &str,
        bindings: &BTreeMap<String, (crate::syntax::Node, String)>,
        prefix: &str,
    ) -> Result<String, crate::SourceError> {
        self.library
            .expression(
                self.source.text(),
                self.source.declarations(),
                crate::model::Vocabulary {
                    types: &self.types,
                    encodings: &self.encodings,
                    data: &self.data,
                },
                node,
                result,
                bindings,
                prefix,
            )
            .map_err(|e| self.source.locate(e))
    }
}

/// A universal type variable or an independent member of a domain.
#[derive(Debug, Clone)]
pub struct Term {
    pub variable: Option<u8>,
    pub domain: TypeSet,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub inputs: Vec<Term>,
    pub results: Vec<Term>,
}

/// A fixed-arity value contract. Properties, successors and structural types
/// need an explicit adapter, rather than silently disappearing from the rule.
#[derive(Debug, Clone)]
pub struct Operation {
    pub name: String,
    pub signature: Result<Signature, String>,
    pub constrained: bool,
    pub primitive: Option<veloc_semantics::BvOp>,
    /// Generated constructor, when the declared storage supports value-only calls.
    pub constructor: Option<String>,
}

impl Definitions {
    pub fn operations(&self) -> impl Iterator<Item = Operation> + '_ {
        self.ops.iter().map(|op| {
            let signature = (|| {
                if op.params.iter().any(|p| p.kind != ParamKind::Value) {
                    return Err(
                        "properties, variadic operands or successors require an adapter".into(),
                    );
                }
                if op.meta.value_only.is_none()
                    || ["ABORT", "TERMINATOR"]
                        .iter()
                        .any(|name| op.traits.contains(*name))
                {
                    return Err("memory or control effects require an adapter".into());
                }
                let inputs = op
                    .signature
                    .operands
                    .patterns()
                    .ok_or("dynamic input signature")?;
                let results = op
                    .signature
                    .results
                    .patterns()
                    .ok_or("dynamic result signature")?;
                let mut variables = BTreeMap::new();
                // A binder may first occur in a result.
                for pattern in inputs.iter().chain(results) {
                    if let Pattern::Bind(id, set) = pattern {
                        variables.insert(*id, set.clone());
                    }
                }
                let term = |pattern: &Pattern| -> Result<Term, String> {
                    let (variable, domain) = match pattern {
                        Pattern::Bind(id, set) => (Some(*id), set.clone()),
                        Pattern::Same(id) => (Some(*id), variables[id].clone()),
                        Pattern::Set(set) => (None, set.clone()),
                        Pattern::Exact(name) => (None, self.types.exact[name].clone()),
                        _ => {
                            return Err(
                                "structural or shape-dependent type requires an adapter".into()
                            );
                        }
                    };
                    Ok(Term { variable, domain })
                };
                Ok(Signature {
                    inputs: inputs.iter().map(term).collect::<Result<_, _>>()?,
                    results: results.iter().map(term).collect::<Result<_, _>>()?,
                })
            })();
            // A primitive describes only the value computation. Never infer a
            // rewrite that drops a memory effect or observable control behavior.
            let primitive = if !["MAY_TRAP", "ABORT", "TERMINATOR"]
                .iter()
                .any(|name| op.traits.contains(*name))
            {
                op.semantics.as_ref().and_then(|s| s.primitive())
            } else {
                None
            };
            let constructor = if signature.is_ok() {
                match &self.storage.strategy {
                    crate::storage::Strategy::Operands(_) => Some(crate::model::mnemonic(&op.name)),
                    crate::storage::Strategy::Packed => None,
                }
            } else {
                None
            };
            Operation {
                name: op.name.clone(),
                signature,
                constrained: !op.constraints.is_empty(),
                constructor,
                primitive,
            }
        })
    }
}
