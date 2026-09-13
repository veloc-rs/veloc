//! Checked, structured output preparation. No Rust artifacts are emitted here.
use std::collections::BTreeMap;

use super::{evaluate, packing};
use crate::model::Definitions;
use crate::storage::Strategy;
use crate::{Error, Generated};

/// A definition unit with all selected-output checks and projections prepared.
/// Construction can fail; generating artifacts from a plan cannot report a
/// definition error. Fields are private so consumers cannot invalidate it.
pub struct Plan {
    pub(super) definitions: Definitions,
    pub(super) output: Output,
}

pub(super) enum Output {
    Packed(Packed),
    Operands,
}

pub(super) struct Packed {
    /// Format indices in operation order, resolved once during preparation.
    pub formats: Vec<usize>,
    pub builders: Vec<Option<packing::Builder>>,
    pub alternatives: Vec<packing::Alternative>,
    pub text: crate::text::Plan,
    pub evaluation: evaluate::Plan,
}

impl Plan {
    pub(crate) fn prepare(definitions: Definitions, source: &str) -> Result<Self, Error> {
        let output = match &definitions.storage.strategy {
            Strategy::Packed => {
                // A query has one concrete input context across all its opcodes.
                // Context-free arms can still participate in that same query.
                let mut contexts = BTreeMap::new();
                for op in &definitions.ops {
                    for (name, expr) in &op.queries {
                        if let Some(ty) = expr.context_type()
                            && let Some(previous) = contexts.insert(name, ty)
                            && previous != ty
                        {
                            return Err(Error::at(
                                source,
                                op.offset,
                                format!("query `{name}` requires incompatible context types"),
                            ));
                        }
                    }
                }
                let indices = definitions
                    .storage
                    .formats
                    .iter()
                    .enumerate()
                    .map(|(index, format)| (format.name.as_str(), index))
                    .collect::<BTreeMap<_, _>>();
                let formats = definitions
                    .ops
                    .iter()
                    .map(|op| indices[op.format.as_str()])
                    .collect::<Vec<_>>();
                let builders = definitions
                    .ops
                    .iter()
                    .map(|op| packing::prepare_builder(op, source))
                    .collect::<Result<_, _>>()?;
                let alternatives = packing::prepare_alternatives(&definitions, &formats, source)?;
                let text =
                    crate::text::Plan::prepare(&definitions, &formats, &alternatives, source)?;
                let evaluation = evaluate::Plan::prepare(&definitions, source)?;
                Output::Packed(Packed {
                    formats,
                    builders,
                    alternatives,
                    text,
                    evaluation,
                })
            }
            Strategy::Operands(_) => {
                if let Some(property) = definitions
                    .properties
                    .iter()
                    .find(|p| !p.constraints.is_empty())
                {
                    return Err(Error::at(
                        source,
                        property.offset,
                        "operand storage does not yet support property validators",
                    ));
                }
                for op in &definitions.ops {
                    if op.constraints.iter().any(|c| !c.type_only)
                        || op.text.is_some()
                        || !op.queries.is_empty()
                        || op.params.iter().any(|p| p.moves)
                        || op.signature_source.is_some()
                    {
                        return Err(Error::at(
                            source,
                            op.offset,
                            "operand storage does not yet support structural constraints, text adapters or ownership interfaces",
                        ));
                    }
                }
                Output::Operands
            }
        };
        Ok(Self {
            definitions,
            output,
        })
    }

    /// Emit reusable artifacts without rechecking or resolving source syntax.
    pub fn generate(&self) -> Generated {
        super::generate(self)
    }

    pub fn definitions(&self) -> &Definitions {
        &self.definitions
    }
}
