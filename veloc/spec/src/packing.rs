//! Typed projections between logical parameters and physical instruction fields.

use crate::model::{Binding, Op};
use crate::storage::{FieldType, Format};

/// Construct physical storage from already typed logical locals.
pub(crate) fn constructor(
    op: &Op,
    format: &Format,
    dfg: &str,
    local: impl Fn(&str) -> String,
) -> String {
    let mut fields = Vec::new();
    for field in &format.fields {
        let value = if matches!(&field.ty, FieldType::Named(ty) if ty == "Opcode") {
            format!("crate::Opcode::{}", op.name)
        } else {
            match &op.packing[&field.name] {
                Binding::Name(name) => {
                    let value = local(name);
                    if field.ty.named("BlockCall") {
                        format!("({value}).as_view()")
                    } else if field.ty.named("ValueList") {
                        format!("&{value}")
                    } else {
                        value
                    }
                }
                Binding::Array(args) => {
                    let args = args
                        .iter()
                        .map(|arg| {
                            let Binding::Name(name) = arg else {
                                unreachable!("checked array binding")
                            };
                            local(name)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{args}]")
                }
                Binding::Pool(name) => {
                    let value = local(name);
                    let ty = field.ty.qualified_type();
                    format!("{ty}::insert(&mut {dfg}, {value})")
                }
                Binding::Table { cases, default } => {
                    format!(
                        "({}).iter().map(crate::BlockCall::as_view).chain(core::iter::once(({}).as_view()))",
                        local(cases),
                        local(default)
                    )
                }
            }
        };
        fields.push(value);
    }
    format!(
        "crate::InstDraft::{}({})",
        crate::storage::constructor_name(&format.name),
        fields.join(", ")
    )
}

/// Recover logical locals from physical values. Records, byte buffers and
/// variadic lists are borrowed; the caller selects its error representation.
pub(crate) fn projections(
    op: &Op,
    format: &Format,
    dfg: &str,
    field: impl Fn(&str) -> String,
    required: impl Fn(String) -> String,
) -> Vec<(String, String)> {
    let mut locals = Vec::new();
    for storage in &format.fields {
        if matches!(&storage.ty, FieldType::Named(ty) if ty == "Opcode") {
            continue;
        }
        let value = field(&storage.name);
        match &op.packing[&storage.name] {
            Binding::Name(name) => {
                locals.push((name.clone(), value));
            }
            Binding::Array(args) => {
                let value = format!("({value})");
                for (index, arg) in args.iter().enumerate() {
                    let Binding::Name(name) = arg else {
                        unreachable!("checked array binding")
                    };
                    locals.push((name.clone(), format!("{value}[{index}]")));
                }
            }
            Binding::Pool(name) => {
                let ty = storage.ty.qualified_type();
                let value = required(format!("{ty}::get({value}, {dfg})"));
                locals.push((name.clone(), value));
            }
            Binding::Table { cases, default } => {
                let split = required(format!("({value}).split_last()"));
                locals.push((cases.clone(), format!("({split}).1")));
                locals.push((default.clone(), format!("({split}).0")));
            }
        }
    }
    locals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_immutable_bytes_are_interned() {
        let source = [
            include_str!("../../mir/defs/formats.ops"),
            include_str!("../../mir/defs/mir.ops"),
        ]
        .join("\n");
        let defs = crate::fixtures::parse(&source).unwrap();
        for (name, logical, pooled) in [
            ("PtrIndex", "imm", false),
            ("LoadStride", "mem", false),
            ("Vconst", "bytes", true),
        ] {
            let op = defs.ops.iter().find(|op| op.name == name).unwrap();
            let format = defs
                .storage
                .formats
                .iter()
                .find(|f| f.name == op.format)
                .unwrap();
            let packed = constructor(op, format, "dfg", str::to_owned);
            assert_eq!(packed.contains("::insert("), pooled, "{packed}");
            let locals = projections(op, format, "dfg", str::to_owned, |value| {
                format!("{value}.ok_or(invalid)?")
            });
            let (_, expr) = locals.iter().find(|(name, _)| name == logical).unwrap();
            assert_eq!(expr.contains("::get("), pooled, "{expr}");
        }
    }

    #[test]
    fn jump_table_projection_splits_default_from_cases() {
        let source = [
            include_str!("../../mir/defs/formats.ops"),
            include_str!("../../mir/defs/mir.ops"),
        ]
        .join("\n");
        let defs = crate::fixtures::parse(&source).unwrap();
        let op = defs.ops.iter().find(|op| op.name == "BrTable").unwrap();
        let format = defs
            .storage
            .formats
            .iter()
            .find(|format| format.name == op.format)
            .unwrap();
        assert!(
            constructor(op, format, "dfg", str::to_owned)
                .contains("chain(core::iter::once((default).as_view()))")
        );
        let locals = projections(op, format, "dfg", str::to_owned, |value| {
            format!("{value}.ok_or(invalid)?")
        });
        assert!(
            locals
                .iter()
                .any(|(name, expr)| name == "cases" && expr.ends_with(").1"))
        );
        assert!(locals.iter().any(|(name, expr)| name == "default"
            && expr.starts_with("(")
            && expr.ends_with(").0")));
    }
}
