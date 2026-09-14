//! Emit operand transfer visitors. CFG edge arguments are handled by the
//! dataflow engine, independently for each successor occurrence.
use crate::model::records::PropertyType;
use crate::model::{Definitions, ParamKind};
use std::collections::BTreeMap;
use std::fmt::Write;

pub(crate) fn generate(defs: &Definitions) -> String {
    let mut groups = BTreeMap::<String, Vec<&str>>::new();
    for op in &defs.ops {
        let edges = op
            .params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Successor | ParamKind::Successors));
        let body = if !op.params.iter().any(|p| p.moves) && !edges {
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
            let projections: BTreeMap<_, _> = crate::model::access::projections(
                op,
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
            body.push_str(&body_for(defs, op, &projections));
            body
        };
        groups.entry(body).or_default().push(&op.name);
    }
    let mut out = String::from(
        "impl crate::InstView<'_> {\n/// Visit non-edge inputs and whether they transfer ownership.\npub(crate) fn try_visit_ownership<E>(&self, mut visit: impl FnMut(crate::Value, bool) -> core::result::Result<(), E>) -> core::result::Result<(), E> {\nmatch self.opcode() {\n",
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
    let moving = defs
        .ops
        .iter()
        .filter(|op| op.params.iter().any(|p| p.moves))
        .map(|op| format!("Self::{}", op.name))
        .collect::<Vec<_>>();
    let query = if moving.is_empty() {
        "false".to_owned()
    } else {
        format!("matches!(self, {})", moving.join(" | "))
    };
    writeln!(
        out,
        "impl crate::Opcode {{ pub const fn transfers_ownership(self) -> bool {{ {query} }} }}"
    )
    .unwrap();
    out
}
/// Ownership semantics only depend on logical parameters and their access paths.
pub(crate) fn body_for(
    defs: &Definitions,
    op: &crate::model::Op,
    projections: &BTreeMap<String, String>,
) -> String {
    let mut body = String::new();
    for p in &op.params {
        let value = &projections[&p.name];
        let consume = p.moves;
        match &p.kind {
            ParamKind::Value => writeln!(body, "visit({value}, {consume})?;").unwrap(),
            ParamKind::Values => writeln!(
                body,
                "for &value in ({value}).iter() {{ visit(value, {consume})?; }}"
            )
            .unwrap(),
            ParamKind::Property(name) => {
                if let Some(record) = defs.data.records.iter().find(|r| r.name == *name) {
                    for field in &record.fields {
                        match &field.ty {
                            PropertyType::Named(_) if field.policy.references.is_operand() => {
                                writeln!(
                                    body,
                                    "visit(({}).{}, false)?;",
                                    value.strip_prefix('*').unwrap_or(value),
                                    field.name
                                )
                                .unwrap()
                            }
                            PropertyType::Optional(_) if field.policy.references.is_operand() => {
                                writeln!(
                                    body,
                                    "if let Some(value) = ({}).{} {{ visit(value, false)?; }}",
                                    value.strip_prefix('*').unwrap_or(value),
                                    field.name
                                )
                                .unwrap()
                            }
                            _ => {}
                        }
                    }
                }
            }
            ParamKind::Successor | ParamKind::Successors => {}
        }
    }

    body
}

pub(crate) fn operand_methods(defs: &Definitions) -> String {
    let crate::storage::Strategy::Operands(storage) = &defs.storage.strategy else {
        unreachable!("operand host")
    };
    let mut out = format!(
        "fn try_visit_ownership<E>(self, mut visit: impl FnMut({}, bool) -> core::result::Result<(), E>) -> core::result::Result<(), E> {{ match self.opcode() {{\n",
        storage.register_rust
    );
    for op in &defs.ops {
        let locals = op.operands().projections(op, |v| {
            format!("({v}).expect(\"validated operand projection\")")
        });
        writeln!(
            out,
            "Some({}::{}) => {{ {} }},",
            storage.opcode,
            op.name,
            body_for(defs, op, &locals)
        )
        .unwrap();
    }
    out.push_str("_ => panic!(\"expected instruction from this opcode set\"), } Ok(()) }\n");
    out
}
