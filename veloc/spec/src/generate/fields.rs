//! Lower logical rule attributes to MIR's shared instruction fields.
use std::collections::BTreeMap;

use crate::{
    Definitions,
    model::{Binding, Op},
};

pub(crate) fn constructor(
    defs: &Definitions,
    op: &Op,
    attributes: &BTreeMap<String, String>,
    opcode: &str,
) -> String {
    let format = defs
        .storage
        .formats
        .iter()
        .find(|f| f.name == op.format)
        .unwrap();
    let args = format
        .fields
        .iter()
        .filter(|f| f.access().is_none())
        .map(|f| match op.bindings().get(&f.name) {
            Some(Binding::Name(param)) => attributes[param].clone(),
            None if f.ty.named("Opcode") => opcode.into(),
            _ => unreachable!("checked expression fields"),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "veloc_mir::InstFields::{}({args})",
        crate::storage::constructor_name(&op.format)
    )
}
