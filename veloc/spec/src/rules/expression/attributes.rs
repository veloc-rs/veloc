//! Attribute adapters come from logical operands and their checked storage
//! projections. Matching, congruence and construction use the same representation.
use super::*;
use crate::model::{ParamKind, expr::Emitter};
use std::fmt::Write;

fn operations(defs: &Definitions) -> Vec<&crate::model::Op> {
    let contracts: BTreeMap<_, _> = defs.operations().map(|op| (op.name.clone(), op)).collect();
    defs.ops
        .iter()
        .filter(|op| {
            (op.semantics.is_some()
                || (contracts[&op.name]
                    .expression
                    .as_ref()
                    .is_some_and(|sig| sig.results.len() == 1)
                    && !op.traits.contains("MAY_TRAP")
                    && op
                        .constraints
                        .iter()
                        .all(|c| c.condition.context_type().is_none())))
                && op
                    .params
                    .iter()
                    .any(|p| matches!(p.kind, ParamKind::Property(_)))
        })
        .collect()
}

pub(in crate::rules) fn emit_attributes(defs: &Definitions) -> String {
    let mut code = String::from(
        "#[allow(unused_variables)]\npub(crate) fn validate_fields(fields: &veloc_mir::InstFields, args: &[Type], results: &[Type]) -> Option<()> {\nlet opcode = fields.opcode();\nopcode.validate_types(args, results).ok()?;\nif !opcode.spec().is_pure() || opcode.transfers_ownership() { return None; }\nmatch opcode {\n",
    );
    for op in operations(defs) {
        let props: BTreeMap<_, _> = op
            .params
            .iter()
            .filter(|p| matches!(p.kind, ParamKind::Property(_)))
            .map(|p| (p.name.clone(), p.name.clone()))
            .collect();
        writeln!(code, "Opcode::{} => {{", op.name).unwrap();
        code.push_str(&crate::storage::compact::bind_attributes(
            op,
            &defs.storage,
            &props,
            "fields",
            "return None;",
        ));
        let emitter = Emitter::types(op, props, "args", "results");
        for c in &op.constraints {
            if !c.type_only && !c.redundant() {
                code.push_str(&c.emit(&emitter, "return None"));
            }
        }
        code.push_str("Some(()) },\n");
    }
    code.push_str("_ => (veloc_mir::InstFields::from_opcode(opcode).as_ref() == Some(fields)).then_some(()),\n} }\n");
    code
}
