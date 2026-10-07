//! Logical parameter access, independent of instruction containers and layouts.
//! Storage resolves these paths once; all consumers use the same lowering.
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) enum Access {
    Field(String),
    Index(Box<Self>, usize),
    Required(Box<Self>),
    SplitLast(Box<Self>, usize),
}

impl Access {
    pub(crate) fn emit(
        &self,
        field: &impl Fn(&str) -> String,
        required: &impl Fn(String) -> String,
    ) -> String {
        let emit = |value: &Self| value.emit(field, required);
        match self {
            Self::Field(name) => field(name),
            Self::Index(value, index) => format!("({})[{index}]", emit(value)),
            Self::Required(value) => required(emit(value)),
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
    field: impl Fn(&str) -> String,
    required: impl Fn(String) -> String,
) -> Vec<(String, String)> {
    op.inputs
        .iter()
        .map(|(name, access)| (name.clone(), access.emit(&field, &required)))
        .collect()
}

pub(crate) type Inputs = BTreeMap<String, Access>;

/// Emit a storage constructor from logical operands, using the checked mapping.
pub(crate) fn constructor(
    op: &super::Op,
    storage: &crate::storage::Storage,
    inputs: &BTreeMap<String, String>,
    opcode: &str,
    writer: &str,
) -> String {
    fn binding(value: &super::Binding, inputs: &BTreeMap<String, String>) -> String {
        match value {
            super::Binding::Name(name) => inputs[name].clone(),
            super::Binding::Array(parts) => format!(
                "[{}]",
                parts
                    .iter()
                    .map(|p| binding(p, inputs))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            super::Binding::Table { .. } => unreachable!("fixed operation has no successors"),
        }
    }
    let format = storage
        .formats
        .iter()
        .find(|f| f.name == op.format)
        .unwrap();
    let fields = format
        .fields
        .iter()
        .map(|f| {
            op.bindings()
                .get(&f.name)
                .map(|b| binding(b, inputs))
                .unwrap_or_else(|| opcode.into())
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{writer}.{}({fields})",
        crate::storage::constructor_name(&op.format)
    )
}
