//! Attribute adapters come from logical operands and their checked storage
//! projections. Matching, congruence and construction use the same representation.
use super::*;
use crate::model::{Binding, ParamKind, expr::Emitter};
use std::fmt::Write;

pub(in crate::rules) fn emit_attributes(defs: &Definitions) -> String {
    let contracts: BTreeMap<_, _> = defs.operations().map(|op| (op.name.clone(), op)).collect();
    let ops: Vec<_> = defs
        .ops
        .iter()
        .filter(|op| {
            contracts[&op.name]
                .expression
                .as_ref()
                .is_some_and(|sig| sig.results.len() == 1)
                && !op.traits.contains("MAY_TRAP")
                && op
                    .constraints
                    .iter()
                    .all(|c| c.condition.context_type().is_none())
                && op
                    .params
                    .iter()
                    .any(|p| matches!(p.kind, ParamKind::Property(_)))
        })
        .collect();
    let mut code = String::from(
        "mod attributes {\n#[allow(unused_imports)] use veloc_mir::*;\n#[allow(unused_imports)] use veloc_mir::inst::*;\n#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\npub(crate) enum Properties { None,\n",
    );
    for op in &ops {
        writeln!(code, "{} {{", op.name).unwrap();
        for p in &op.params {
            if let ParamKind::Property(ty) = &p.kind {
                writeln!(code, "{}: {ty},", p.name).unwrap();
            }
        }
        code.push_str("},\n");
    }
    code.push_str("}\nimpl Properties {\n#[allow(unused_variables)]\npub(crate) fn read(view: InstView<'_>) -> Self { match view.opcode() {\n");
    for op in &ops {
        let fields = defs
            .storage
            .formats
            .iter()
            .find(|f| f.name == op.format)
            .unwrap();
        let names = fields
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(code, "Opcode::{} => {{ let InstView::{} {{ {names} }} = view else {{ unreachable!() }}; Self::{} {{", op.name, op.format, op.name).unwrap();
        for p in &op.params {
            if matches!(p.kind, ParamKind::Property(_)) {
                let value = op.inputs[&p.name].emit(&|name| name.into(), &|value| {
                    format!("{value}.expect(\"checked property\")")
                });
                writeln!(code, "{}: {value},", p.name).unwrap();
            }
        }
        code.push_str("} },\n");
    }
    code.push_str("_ => Self::None,\n} }\n");
    code.push_str("#[allow(unused_variables)]\npub(crate) fn validate(self, opcode: Opcode, args: &[Type], results: &[Type]) -> Option<()> {\nopcode.validate_types(args, results).ok()?;\nif !opcode.spec().is_pure() || opcode.transfers_ownership() { return None; }\nmatch (opcode, self) {\n");
    for op in &ops {
        let props: Vec<_> = op
            .params
            .iter()
            .filter(|p| matches!(p.kind, ParamKind::Property(_)))
            .map(|p| p.name.clone())
            .collect();
        writeln!(
            code,
            "(Opcode::{}, Self::{} {{ {} }}) => {{",
            op.name,
            op.name,
            props.join(", ")
        )
        .unwrap();
        let emitter = Emitter::types(
            op,
            props.iter().map(|p| (p.clone(), p.clone())).collect(),
            "args",
            "results",
        );
        for c in &op.constraints {
            if !c.type_only && !c.redundant() {
                code.push_str(&c.emit(&emitter, "return None"));
            }
        }
        code.push_str("Some(()) },\n");
    }
    let names = ops
        .iter()
        .map(|op| format!("Opcode::{}", op.name))
        .collect::<Vec<_>>()
        .join(" | ");
    if !names.is_empty() {
        writeln!(code, "({names}, _) => None,").unwrap();
    }
    code.push_str("(_, Self::None) => Some(()),\n#[allow(unreachable_patterns)] _ => None,\n} }\n");
    code.push_str("pub(crate) fn write(self, opcode: Opcode, args: &[Value], writer: InstWriter<'_>) -> Inst { match (opcode, self) {\n");
    for op in &ops {
        let props: Vec<_> = op
            .params
            .iter()
            .filter(|p| matches!(p.kind, ParamKind::Property(_)))
            .map(|p| p.name.clone())
            .collect();
        let mut inputs: BTreeMap<String, String> =
            props.iter().map(|p| (p.clone(), p.clone())).collect();
        for (i, p) in op
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Value)
            .enumerate()
        {
            inputs.insert(p.name.clone(), format!("args[{i}]"));
        }
        fn binding(b: &Binding, inputs: &BTreeMap<String, String>) -> String {
            match b {
                Binding::Name(name) => inputs[name].clone(),
                Binding::Array(parts) => format!(
                    "[{}]",
                    parts
                        .iter()
                        .map(|p| binding(p, inputs))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                Binding::Table { .. } => unreachable!("expression has no successors"),
            }
        }
        let format = defs
            .storage
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
                    .map(|b| binding(b, &inputs))
                    .unwrap_or_else(|| "opcode".into())
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            code,
            "(Opcode::{}, Self::{} {{ {} }}) => writer.{}({fields}),",
            op.name,
            op.name,
            props.join(", "),
            crate::storage::constructor_name(&op.format)
        )
        .unwrap();
    }
    code.push_str("(_, Self::None) => writer.from_values(opcode, args).expect(\"checked value operation\"),\n_ => unreachable!(\"checked operation properties\"),\n} }\n} }\npub(crate) use attributes::Properties;\n");
    code
}
