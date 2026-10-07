//! Checked, structured output preparation. No Rust artifacts are emitted here.
use std::collections::BTreeMap;

use super::{evaluate, packing};
use crate::model::Definitions;
use crate::storage::Strategy;
use crate::{Artifacts, Error};

/// A definition unit with all selected-output checks and projections prepared.
/// Construction can fail; generating artifacts from a plan cannot report a
/// definition error. Fields are private so consumers cannot invalidate it.
pub struct Plan {
    pub(super) definitions: Definitions,
    pub(super) output: Output,
}

pub(super) enum Output {
    Packed(Packed),
    Operands(crate::text::Plan),
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
        let layouts = match &definitions.storage.strategy {
            Strategy::Operands(storage) => storage.record_names(),
            Strategy::Packed => Vec::new(),
        };
        definitions.data.validate_sequences(
            &layouts,
            crate::model::metadata::record_type(&definitions.ops),
            source,
        )?;
        let output = match &definitions.storage.strategy {
            Strategy::Packed => {
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
                let evaluation = evaluate::Plan::prepare(&definitions);
                Output::Packed(Packed {
                    formats,
                    builders,
                    alternatives,
                    text,
                    evaluation,
                })
            }
            Strategy::Operands(storage) => {
                if let Some(payloads) = &storage.payloads {
                    for op in &definitions.ops {
                        let fields: Vec<_> = op
                            .operands()
                            .members
                            .iter()
                            .filter(|m| {
                                m.domain == crate::storage::operands::Domain::Attribute
                                    && m.binding.is_some()
                            })
                            .map(|m| {
                                (
                                    m.field.codec.as_ref().unwrap().rsplit_once("::").unwrap().1,
                                    String::new(),
                                    m.field.shape,
                                )
                            })
                            .collect();
                        payloads
                            .construct(&fields, "writer", &payloads.rust)
                            .map_err(|e| Error::at(source, op.offset, e))?;
                    }
                }
                Output::Operands(crate::text::Plan::prepare(&definitions, &[], &[], source)?)
            }
        };
        Ok(Self {
            definitions,
            output,
        })
    }

    /// Emit reusable artifacts without rechecking or resolving source syntax.
    pub fn generate(&self) -> Artifacts {
        crate::Emit::ALL
            .iter()
            .copied()
            .filter(|&kind| self.supports(kind))
            .map(|kind| (kind, self.emit(kind)))
            .collect()
    }

    pub(crate) fn supports(&self, artifact: crate::Emit) -> bool {
        use crate::Emit;
        match self.output {
            Output::Operands(_) => matches!(
                artifact,
                Emit::Types
                    | Emit::TypeRules
                    | Emit::Instructions
                    | Emit::Checks
                    | Emit::Semantics
                    | Emit::TextParser
                    | Emit::TextPrinter
            ),
            Output::Packed(_) => matches!(
                artifact,
                Emit::Types
                    | Emit::TypeRules
                    | Emit::Opcodes
                    | Emit::Instructions
                    | Emit::Builders
                    | Emit::Validator
                    | Emit::Checks
                    | Emit::Evaluation
                    | Emit::Semantics
                    | Emit::TextParser
                    | Emit::TextPrinter
            ),
        }
    }

    pub(crate) fn emit(&self, artifact: crate::Emit) -> String {
        super::emit(self, artifact)
    }

    pub fn definitions(&self) -> &Definitions {
        &self.definitions
    }
}
