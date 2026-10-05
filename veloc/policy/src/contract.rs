use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The pass owns feature meaning, action meaning and the lifetime of a session.
/// Action zero always delegates to the pass's ordinary heuristic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DecisionSchema {
    pub name: &'static str,
    pub version: u32,
    pub features: &'static [&'static str],
    pub actions: &'static [&'static str],
    pub scope: &'static str,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Contract {
    version: u32,
    features: Vec<String>,
    actions: Vec<String>,
    scope: String,
}

impl Contract {
    pub fn matches(&self, schema: &DecisionSchema) -> bool {
        self.version == schema.version
            && self
                .features
                .iter()
                .map(String::as_str)
                .eq(schema.features.iter().copied())
            && self
                .actions
                .iter()
                .map(String::as_str)
                .eq(schema.actions.iter().copied())
            && self.scope == schema.scope
    }
}

/// Target properties are supplied by the embedding compiler. Labels identify
/// experiments and optional deployment restrictions; only numbers enter a model.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub labels: BTreeMap<String, String>,
    pub features: BTreeMap<String, f32>,
}

/// Named fields and their serialized order are declared in the owning pass.
#[macro_export]
macro_rules! feature_set {
    ($vis:vis $name:ident { $($field:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Default)]
        $vis struct $name { $(pub $field: f32),+ }
        impl $name {
            pub const NAMES: &'static [&'static str] = &[$(stringify!($field)),+];
            pub fn values(self) -> [f32; Self::NAMES.len()] { [$(self.$field),+] }
        }
    };
}
