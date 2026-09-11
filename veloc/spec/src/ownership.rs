//! Compile operand transfer contracts. CFG edge arguments are handled by the
//! dataflow engine, independently for each successor occurrence.
use crate::model::{Definitions, ParamKind};
use crate::records::PropertyType;
use std::collections::BTreeMap;
use std::fmt::Write;

pub(crate) fn generate(defs: &Definitions) -> String {
    let mut groups = BTreeMap::<String, Vec<&str>>::new();
    for op in &defs.ops {
        let edges = op
            .params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Successor | ParamKind::Successors));
        let body = if op.moves.is_empty() && !edges {
            "self.try_visit_operands(|value| visit(value, false))?;".into()
        } else {
            let format = defs
                .storage
                .formats
                .iter()
                .find(|f| f.name == op.format)
                .unwrap();
            let fields = format
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| format!("{}: _f{i}", f.name))
                .collect::<Vec<_>>()
                .join(", ");
            let projections: BTreeMap<_, _> = crate::packing::projections(
                op,
                format,
                "dfg",
                |name| {
                    format!(
                        "*_f{}",
                        format.fields.iter().position(|f| f.name == name).unwrap()
                    )
                },
                |v| format!("{v}.expect(\"validated operand projection\")"),
            )
            .into_iter()
            .collect();
            let mut body = format!(
                "let Self::{} {{ {fields} }} = self else {{ unreachable!(\"validated transfer storage\") }};\n",
                format.name
            );
            for p in &op.params {
                let value = &projections[&p.name];
                let consume = op.moves.contains(&p.name);
                match &p.kind {
                    ParamKind::Value => writeln!(body, "visit({value}, {consume})?;").unwrap(),
                    ParamKind::Values => writeln!(
                        body,
                        "for &value in ({value}).iter() {{ visit(value, {consume})?; }}"
                    )
                    .unwrap(),
                    ParamKind::Property(name) => {
                        if let Some(record) = defs.storage.records.iter().find(|r| r.name == *name)
                        {
                            for field in &record.fields {
                                match &field.ty {
                                    PropertyType::Named(ty) if ty == "Value" => writeln!(
                                        body,
                                        "visit(({}).{}, false)?;",
                                        value.strip_prefix('*').unwrap_or(value),
                                        field.name
                                    )
                                    .unwrap(),
                                    PropertyType::Optional(ty) if ty == "Value" => writeln!(
                                        body,
                                        "if let Some(value) = ({}).{} {{ visit(value, false)?; }}",
                                        value.strip_prefix('*').unwrap_or(value),
                                        field.name
                                    )
                                    .unwrap(),
                                    _ => {}
                                }
                            }
                        }
                    }
                    ParamKind::Successor | ParamKind::Successors => {}
                }
            }
            body
        };
        groups.entry(body).or_default().push(&op.name);
    }
    let mut out = String::from(
        "impl crate::InstructionView<'_> {\n/// Visit non-edge inputs and whether they transfer ownership.\npub(crate) fn try_visit_ownership<E>(&self, mut visit: impl FnMut(crate::Value, bool) -> core::result::Result<(), E>) -> core::result::Result<(), E> {\nmatch self.opcode() {\n",
    );
    for (body, ops) in groups {
        let arms = ops
            .iter()
            .map(|op| format!("crate::Opcode::{op}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(out, "{arms} => {{ {body} }},").unwrap();
    }
    out.push_str("}\nOk(())\n}\n}\n");
    out
}
