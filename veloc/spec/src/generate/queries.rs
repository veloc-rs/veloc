//! Emit typed interface queries from the shared expression model.
use crate::model::{Definitions, expr::Emitter};
use std::fmt::Write;
pub(crate) fn generate(defs: &Definitions, formats: &[usize]) -> String {
    if defs.expressions.interfaces.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (name, interface) in &defs.expressions.interfaces {
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{"
        )
        .unwrap();
        for (field, ty) in &interface.fields {
            writeln!(out, "pub {field}: {},", ty.rust(&defs.data.rust)).unwrap();
        }
        out.push_str("}\n");
        let context = defs
            .ops
            .iter()
            .filter_map(|op| op.interfaces.get(name))
            .find_map(|expr| expr.context_type())
            .map(|ty| format!("&{ty}"));
        let param = context
            .map(|ty| format!(", _context: {ty}"))
            .unwrap_or_default();
        writeln!(out,"impl {name} {{ pub fn query(view: &InstView<'_>, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]{param}) -> Option<Self> {{ let _ = (dfg, results); match view.opcode() {{").unwrap();
        for (op, &format) in defs.ops.iter().zip(formats) {
            let Some(expr) = op.interfaces.get(name) else {
                continue;
            };
            let format = &defs.storage.formats[format];
            let fields = format
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| format!("{}: _f{i}", f.name))
                .collect::<Vec<_>>()
                .join(", ");
            let locals = crate::generate::packing::projections(
                op,
                format,
                "dfg",
                |name| {
                    format!(
                        "*_f{}",
                        format.fields.iter().position(|f| f.name == name).unwrap()
                    )
                },
                |v| format!("{v}?"),
            )
            .into_iter()
            .collect();
            writeln!(out,"crate::Opcode::{} => {{ let InstView::{} {{ {fields} }} = view else {{ return None; }}; Some({}) }},",op.name,format.name,Emitter::query(locals).term(expr)).unwrap();
        }
        out.push_str("_ => None, } } }\n");
    }
    out
}
