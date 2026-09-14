//! Logical parameter access, independent of instruction containers and layouts.
//! Storage resolves these paths once; all consumers use the same lowering.
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) enum Access {
    Field(String),
    Index(Box<Self>, usize),
    Required(Box<Self>),
    Pool { ty: String, key: Box<Self> },
    SplitLast(Box<Self>, usize),
}

impl Access {
    pub(crate) fn emit(
        &self,
        dfg: &str,
        field: &impl Fn(&str) -> String,
        required: &impl Fn(String) -> String,
    ) -> String {
        let emit = |value: &Self| value.emit(dfg, field, required);
        match self {
            Self::Field(name) => field(name),
            Self::Index(value, index) => format!("({})[{index}]", emit(value)),
            Self::Required(value) => required(emit(value)),
            Self::Pool { ty, key } => required(format!("{ty}::get({}, {dfg})", emit(key))),
            Self::SplitLast(value, part) => {
                format!(
                    "({}).{part}",
                    required(format!("({}).split_last()", emit(value)))
                )
            }
        }
    }
}

pub(crate) fn projections(
    op: &super::Op,
    dfg: &str,
    field: impl Fn(&str) -> String,
    required: impl Fn(String) -> String,
) -> Vec<(String, String)> {
    op.inputs
        .iter()
        .map(|(name, access)| (name.clone(), access.emit(dfg, &field, &required)))
        .collect()
}

pub(crate) type Inputs = BTreeMap<String, Access>;
