//! Emit named struct queries from the shared expression model.
use crate::model::{Definitions, expr::Emitter};
use std::fmt::Write;
pub(crate) enum Host<'a> {
    Packed(&'a [usize]),
    Operands,
}

/// One query compiler. Hosts only supply dispatch syntax and physical reads.
pub(crate) fn generate(defs: &Definitions, host: Host<'_>) -> String {
    let mut out = String::new();
    for (method, name) in &defs.expressions.queries {
        let context = defs
            .ops
            .iter()
            .filter_map(|op| op.queries.get(method))
            .find_map(|expr| expr.context_type());
        let param = context
            .map(|ty| format!(", _context: &{ty}"))
            .unwrap_or_default();
        match host {
            Host::Packed(_) => writeln!(out, "impl crate::Inst {{ pub fn {method}(self, dfg: &crate::dfg::DataFlowGraph{param}) -> Option<{name}> {{ match dfg.opcode(self) {{").unwrap(),
            Host::Operands => writeln!(out, "fn {method}(self{param}) -> Option<{name}> {{ match self.opcode() {{").unwrap(),
        }
        for (index, op) in defs.ops.iter().enumerate() {
            let Some(expr) = op.queries.get(method) else {
                continue;
            };
            let (emitter, setup, pattern) = match host {
                Host::Packed(formats) => {
                    let format = &defs.storage.formats[formats[index]];
                    let fields = format
                        .fields
                        .iter()
                        .enumerate()
                        .map(|(i, f)| format!("{}: _f{i}", f.name))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let locals = crate::model::access::projections(
                        op,
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
                    let emitter = Emitter::values(locals, "dfg", "dfg.inst_results(self)");
                    let setup = format!(
                        "let view = dfg.inst(self); let InstView::{} {{ {fields} }} = &view else {{ unreachable!(\"opcode and storage layout disagree\"); }};",
                        format.name
                    );
                    (emitter, setup, format!("crate::Opcode::{}", op.name))
                }
                Host::Operands => {
                    let crate::storage::Strategy::Operands(storage) = &defs.storage.strategy else {
                        unreachable!("operand host")
                    };
                    let emitter = Emitter::values(
                        op.operands().projections(op, |v| format!("({v})?")),
                        "self",
                        "self.results()",
                    );
                    (
                        emitter,
                        String::new(),
                        format!("Some({}::{})", storage.opcode, op.name),
                    )
                }
            };
            let value = emitter.term(expr);
            writeln!(out, "{pattern} => {{").unwrap();
            if emitter.storage_used.get() {
                out.push_str(&setup);
            }
            writeln!(out, "Some({value}) }},").unwrap();
        }
        out.push_str("_ => None, } }\n");
        if matches!(host, Host::Packed(_)) {
            out.push_str("}\n");
        }
    }
    out
}
