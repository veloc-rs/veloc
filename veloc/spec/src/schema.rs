//! Read-only logical contracts for rule compilers. No runtime IR or storage ABI.
use std::collections::BTreeMap;

use crate::Definitions;
use crate::model::{ParamKind, Pattern};
pub use crate::types::TypeSet;

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
                    crate::storage::Strategy::Operands(storage) => {
                        Some(format!("build_{}", storage.mnemonic(&op.name)))
                    }
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
