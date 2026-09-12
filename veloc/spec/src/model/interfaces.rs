//! Emit typed interface queries from the shared expression model.
use super::Definitions;
use std::fmt::Write;
pub(crate) fn generate(defs: &Definitions, formats: &[usize]) -> String {
    if defs.expressions.interfaces.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "/// A statically dispatched projection over an instruction's logical fields.\npub trait InstructionQuery: Sized { fn query(view: &InstructionView<'_>, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]) -> Option<Self>; }\nimpl InstructionView<'_> { pub fn query<Q: InstructionQuery>(&self, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]) -> Option<Q> { Q::query(self, dfg, results) } }\n",
    );
    for (name, interface) in &defs.expressions.interfaces {
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{"
        )
        .unwrap();
        for (field, ty) in &interface.fields {
            writeln!(out, "pub {field}: {},", ty.rust()).unwrap();
        }
        out.push_str("}\n");
        writeln!(out,"impl InstructionQuery for {name} {{ fn query(view: &InstructionView<'_>, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]) -> Option<Self> {{ let _ = (dfg, results); match view.opcode() {{").unwrap();
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
            let host = if expr.uses_host() {
                "let _host = crate::host::Context::new(dfg);"
            } else {
                ""
            };
            writeln!(out,"crate::Opcode::{} => {{ let InstructionView::{} {{ {fields} }} = view else {{ return None; }}; {host} Some({}) }},",op.name,format.name,super::expr::Emitter::query(locals).term(expr)).unwrap();
        }
        out.push_str("_ => None, } } }\n");
    }
    out
}
