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
                    if matches!(&field.ty, FieldType::Named(ty) if ty == "ValueList") {
                        format!("{dfg}.make_value_list(&{value})")
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
                    if matches!(field.ty, FieldType::List(_)) {
                        format!("{dfg}.make_value_list(&[{args}])")
                    } else {
                        format!("[{args}]")
                    }
                }
                Binding::Pool(name) => {
                    let value = local(name);
                    let ty = field.ty.qualified_type();
                    format!("<{ty} as crate::dfg::PoolKey>::insert(&mut {dfg}, {value})")
                }
                Binding::Table { cases, default } => {
                    format!(
                        "{dfg}.make_jump_table(&{}, {})",
                        local(cases),
                        local(default)
                    )
                }
            }
        };
        fields.push(if value == field.name {
            value
        } else {
            format!("{}: {value}", field.name)
        });
    }
    if fields.is_empty() {
        format!("crate::InstructionData::{}", format.name)
    } else {
        format!(
            "crate::InstructionData::{} {{ {} }}",
            format.name,
            fields.join(", ")
        )
    }
}

/// Recover logical locals from physical values. Records, byte buffers and
/// variadic lists are borrowed; the caller selects its error representation.
pub(crate) fn projections(
    op: &Op,
    format: &Format,
    dfg: &str,
    field: impl Fn(&str) -> String,
    missing: &str,
) -> Vec<(String, String)> {
    let mut locals = Vec::new();
    for storage in &format.fields {
        if matches!(&storage.ty, FieldType::Named(ty) if ty == "Opcode") {
            continue;
        }
        let value = field(&storage.name);
        match &op.packing[&storage.name] {
            Binding::Name(name) => {
                let value = if matches!(&storage.ty, FieldType::Named(ty) if ty == "ValueList") {
                    format!("{dfg}.get_value_list({value})")
                } else {
                    value
                };
                locals.push((name.clone(), value));
            }
            Binding::Array(args) => {
                let value = if matches!(storage.ty, FieldType::List(_)) {
                    format!("{dfg}.get_value_list({value})")
                } else {
                    format!("({value})")
                };
                for (index, arg) in args.iter().enumerate() {
                    let Binding::Name(name) = arg else {
                        unreachable!("checked array binding")
                    };
                    locals.push((name.clone(), format!("{value}[{index}]")));
                }
            }
            Binding::Pool(name) => {
                let ty = storage.ty.qualified_type();
                let value =
                    format!("<{ty} as crate::dfg::PoolKey>::get({value}, {dfg}).ok_or({missing})?");
                locals.push((name.clone(), value));
            }
            Binding::Table { cases, default } => {
                let split =
                    format!("{dfg}.jump_table_targets({value}).split_last().ok_or({missing})?");
                locals.push((cases.clone(), format!("({split}).1")));
                locals.push((default.clone(), format!("*({split}).0")));
            }
        }
    }
    locals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_pool_projections_borrow_and_constructors_intern_whole_records() {
        let source = [
            include_str!("../../mir/defs/formats.ops"),
            include_str!("../../mir/defs/mir.ops"),
        ]
        .join("\n");
        let defs = crate::model::parse(&source).unwrap();
        for (name, logical, key) in [
            ("PtrIndex", "imm", "PtrIndexImmId"),
            ("LoadStride", "mem", "VectorMemExtId"),
            ("Vconst", "bytes", "ConstantPoolId"),
        ] {
            let op = defs.ops.iter().find(|op| op.name == name).unwrap();
            let format = defs
                .storage
                .formats
                .iter()
                .find(|format| format.name == op.format)
                .unwrap();
            let packed = constructor(op, format, "dfg", str::to_owned);
            assert!(
                packed.contains(&format!(
                    "<crate::inst::{key} as crate::dfg::PoolKey>::insert(&mut dfg, {logical})"
                )),
                "{packed}"
            );
            let locals = projections(op, format, "dfg", str::to_owned, "invalid");
            let (_, expr) = locals.iter().find(|(name, _)| name == logical).unwrap();
            assert!(
                expr.contains(&format!(
                    "<crate::inst::{key} as crate::dfg::PoolKey>::get("
                )),
                "{expr}"
            );
            assert!(expr.contains("ok_or(invalid)?"), "{expr}");
            assert!(!expr.contains("clone"), "{expr}");
            assert!(!expr.contains("ConstantPoolData"), "{expr}");
        }
    }

    #[test]
    fn jump_table_projection_splits_default_from_cases() {
        let source = [
            include_str!("../../mir/defs/formats.ops"),
            include_str!("../../mir/defs/mir.ops"),
        ]
        .join("\n");
        let defs = crate::model::parse(&source).unwrap();
        let op = defs.ops.iter().find(|op| op.name == "BrTable").unwrap();
        let format = defs
            .storage
            .formats
            .iter()
            .find(|format| format.name == op.format)
            .unwrap();
        assert!(
            constructor(op, format, "dfg", str::to_owned)
                .contains("dfg.make_jump_table(&cases, default)")
        );
        let locals = projections(op, format, "dfg", str::to_owned, "invalid");
        assert!(
            locals
                .iter()
                .any(|(name, expr)| name == "cases" && expr.ends_with(").1"))
        );
        assert!(locals.iter().any(|(name, expr)| name == "default"
            && expr.starts_with("*(")
            && expr.ends_with(").0")));
    }
}
