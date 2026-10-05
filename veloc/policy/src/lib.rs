//! Versioned decision contracts and offline-trained neural policies.
//! Passes own legality and scope; models only replace profitability decisions.
mod contract;
mod network;

use contract::Contract;
pub use contract::{Context, DecisionSchema};
use network::{Network, Workspace};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    io::{Read, Write},
    path::Path,
    sync::Mutex,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const FORMAT_VERSION: u32 = 2;
// A file-size limit protects loading; it does not prescribe network topology.
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Advisor {
    Fixed { action: usize },
    Probe { cases: Vec<Intervention> },
    Network(Network),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intervention {
    features: Vec<f32>,
    action: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head<A> {
    contract: Contract,
    /// Applied identically to networks and experiment controls, per pass scope.
    max_deviations: Option<usize>,
    advisor: A,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document<A> {
    version: u32,
    context_features: Vec<String>,
    /// Optional experiment/deployment restrictions, never model inputs.
    requires: BTreeMap<String, Vec<String>>,
    decisions: BTreeMap<String, Head<A>>,
}

#[derive(Debug, Serialize)]
struct Observation {
    decision: &'static str,
    features: Vec<f32>,
    baseline: usize,
    action: usize,
}

/// Weights and context are immutable and shareable across compilation workers.
#[derive(Debug)]
pub struct Policy {
    context: Context,
    context_values: Vec<f32>,
    schemas: BTreeMap<&'static str, DecisionSchema>,
    heads: BTreeMap<String, Head<Advisor>>,
    observations: Option<Mutex<Vec<Observation>>>,
}

impl Policy {
    pub fn new(
        context: Context,
        schemas: impl IntoIterator<Item = DecisionSchema>,
    ) -> Result<Self> {
        if context.features.values().any(|x| !x.is_finite()) {
            return Err("non-finite policy context".into());
        }
        let mut registry = BTreeMap::new();
        for schema in schemas {
            if schema.actions.is_empty() || registry.insert(schema.name, schema).is_some() {
                return Err("empty actions or duplicate decision name".into());
            }
        }
        Ok(Self {
            context,
            context_values: Vec::new(),
            schemas: registry,
            heads: BTreeMap::new(),
            observations: None,
        })
    }

    pub fn with_model(mut self, path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(MAX_MODEL_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_MODEL_BYTES {
            return Err("policy exceeds 64 MiB".into());
        }
        // Packed deployments contain networks directly, avoiding intermediate
        // buffers for Serde's tagged experiment variants on the startup path.
        let document: Document<Advisor> = match bytes.strip_prefix(b"VLPM\x02") {
            Some(payload) => {
                let packed: Document<Network> = rmp_serde::from_slice(payload)?;
                Document {
                    version: packed.version,
                    context_features: packed.context_features,
                    requires: packed.requires,
                    decisions: packed
                        .decisions
                        .into_iter()
                        .map(|(name, head)| {
                            (
                                name,
                                Head {
                                    contract: head.contract,
                                    max_deviations: head.max_deviations,
                                    advisor: Advisor::Network(head.advisor),
                                },
                            )
                        })
                        .collect(),
                }
            }
            None => serde_json::from_slice(&bytes)?,
        };
        if document.version != FORMAT_VERSION {
            return Err("unsupported policy format (expected version 2)".into());
        }
        for (label, allowed) in &document.requires {
            if !self
                .context
                .labels
                .get(label)
                .is_some_and(|value| allowed.contains(value))
            {
                return Err(format!("policy does not support context label {label}").into());
            }
        }
        self.context_values = document
            .context_features
            .iter()
            .map(|name| {
                self.context
                    .features
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("missing policy context feature {name}").into())
            })
            .collect::<Result<_>>()?;
        for (name, head) in &document.decisions {
            let schema = self
                .schemas
                .get(name.as_str())
                .ok_or_else(|| format!("unknown policy decision {name}"))?;
            if !head.contract.matches(schema) {
                return Err(format!("policy contract mismatch for {name}").into());
            }
            let valid_action = |action: usize| -> Result<()> {
                if action < schema.actions.len() {
                    Ok(())
                } else {
                    Err("invalid policy action".into())
                }
            };
            match &head.advisor {
                Advisor::Fixed { action } => valid_action(*action)?,
                Advisor::Probe { cases } => {
                    for case in cases {
                        valid_action(case.action)?;
                        if case.features.len() != schema.features.len()
                            || case.features.iter().any(|x| !x.is_finite())
                        {
                            return Err("invalid probe features".into());
                        }
                    }
                }
                Advisor::Network(network) => network.validate(
                    schema.features.len() + self.context_values.len(),
                    schema.actions.len(),
                )?,
            }
        }
        self.heads = document.decisions;
        Ok(self)
    }

    pub fn with_observations(mut self) -> Self {
        self.observations = Some(Mutex::new(Vec::new()));
        self
    }

    pub fn enabled(&self, schema: &DecisionSchema) -> bool {
        self.heads.contains_key(schema.name) || self.observations.is_some()
    }

    /// Start the scope declared by the pass, e.g. one caller or one function.
    /// Sessions have independent scratch storage and deviation limits.
    pub fn session(&self, schema: &'static DecisionSchema) -> Session<'_> {
        let registered = self
            .schemas
            .get(schema.name)
            .expect("unregistered policy decision");
        assert_eq!(registered, schema);
        let head = self.heads.get(schema.name);
        let work = match head.map(|h| &h.advisor) {
            Some(Advisor::Network(network)) => network.workspace(),
            _ => Workspace::default(),
        };
        Session {
            policy: self,
            schema,
            head,
            state: RefCell::new(SessionState {
                work,
                deviations: 0,
            }),
        }
    }

    pub fn write_observations(&self, path: &Path) -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({
                "version": FORMAT_VERSION, "context": self.context,
                "decisions": self.schemas.iter().map(|(name, schema)| (*name, serde_json::json!({
                    "version": schema.version, "features": schema.features,
                    "actions": schema.actions, "scope": schema.scope,
                }))).collect::<BTreeMap<_, _>>(),
            }),
        )?;
        writeln!(out)?;
        if let Some(observations) = &self.observations {
            for observation in observations.lock().unwrap().iter() {
                serde_json::to_writer(&mut out, observation)?;
                writeln!(out)?;
            }
        }
        out.flush()
    }
}

struct SessionState {
    work: Workspace,
    deviations: usize,
}

/// Pass-local evaluator. No allocation or locking during untraced decisions.
pub struct Session<'a> {
    policy: &'a Policy,
    schema: &'static DecisionSchema,
    head: Option<&'a Head<Advisor>>,
    state: RefCell<SessionState>,
}

impl Session<'_> {
    /// Reuse inference storage when beginning the next scope of the same pass.
    pub fn reset_scope(&self) {
        self.state.borrow_mut().deviations = 0;
    }

    pub fn wants_features(&self) -> bool {
        self.policy.observations.is_some()
            || self.head.is_some_and(|head| {
                head.max_deviations
                    .is_none_or(|limit| self.state.borrow().deviations < limit)
            })
    }

    /// `baseline` is the action that the heuristic would select at this site.
    /// Returning zero asks the caller to use that heuristic, as usual.
    pub fn choose(&self, features: &[f32], baseline: usize) -> usize {
        assert_eq!(features.len(), self.schema.features.len());
        assert!(baseline < self.schema.actions.len());
        let mut state = self.state.borrow_mut();
        let action = self
            .head
            .filter(|head| {
                head.max_deviations
                    .is_none_or(|limit| state.deviations < limit)
            })
            .map_or(0, |head| match &head.advisor {
                Advisor::Fixed { action } => *action,
                Advisor::Probe { cases } => cases
                    .iter()
                    .find(|case| case.features == features)
                    .map_or(0, |case| case.action),
                Advisor::Network(network) => network.choose(
                    features.iter().chain(&self.policy.context_values).copied(),
                    &mut state.work,
                ),
            });
        state.deviations += usize::from(action != 0 && action != baseline);
        if let Some(observations) = &self.policy.observations {
            observations.lock().unwrap().push(Observation {
                decision: self.schema.name,
                features: features.into(),
                baseline,
                action,
            });
        }
        action
    }
}
