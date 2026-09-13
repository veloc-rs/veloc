//! Emit named struct queries from the shared expression model.
use crate::model::{Definitions, expr::Emitter};
use std::fmt::Write;
pub(crate) fn generate(defs: &Definitions, formats: &[usize]) -> String {
    if defs.expressions.queries.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (method, name) in &defs.expressions.queries {
        let context = defs
            .ops
            .iter()
            .filter_map(|op| op.queries.get(method))
            .find_map(|expr| expr.context_type())
            .map(|ty| format!("&{ty}"));
        let param = context
            .map(|ty| format!(", _context: {ty}"))
            .unwrap_or_default();
        writeln!(out,"impl crate::Inst {{ pub fn {method}(self, dfg: &crate::dfg::DataFlowGraph{param}) -> Option<{name}> {{ match dfg.opcode(self) {{").unwrap();
        for (op, &format) in defs.ops.iter().zip(formats) {
            let Some(expr) = op.queries.get(method) else {
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
            let mut emitter = Emitter::query(locals);
            // Result information is fetched only when the projection uses it.
            emitter.results = "dfg.inst_results(self)";
            let value = emitter.term(expr);
            writeln!(out, "crate::Opcode::{} => {{", op.name).unwrap();
            if emitter.storage_used.get() {
                writeln!(out, "let view = dfg.inst(self); let InstView::{} {{ {fields} }} = &view else {{ unreachable!(\"opcode and storage layout disagree\"); }};", format.name).unwrap();
            }
            writeln!(out, "Some({value}) }},").unwrap();
        }
        out.push_str("_ => None, } } }\n");
    }
    out
}
