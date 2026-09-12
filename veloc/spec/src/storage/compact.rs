//! Compact persistent fields; drafts and public views keep their logical types.
use super::generate::{construct, read_field, record, stored_type};
use super::{Layout, OpcodeSource};
use crate::model::records::{PropertyType, RecordDef};
use std::fmt::Write;

// Unknown or large properties stay out of line. This is a storage policy, not
// a restriction on the logical schema. Byte arrays avoid padding for 64-bit data.
fn size(ty: &str) -> Option<usize> {
    Some(match ty {
        "Opcode" | "IntCC" | "FloatCC" | "bool" | "u8" => 1,
        "MemFlags" | "Intrinsic" => 2,
        "u32" | "i32" | "FuncId" | "SigId" | "ConstantPoolId" => 4,
        "u64" => 8,
        "Int" | "Float" => 9,
        "VectorConst" => 11,
        _ => return None,
    })
}

fn last_group(layout: &Layout, records: &[RecordDef]) -> Option<usize> {
    layout.fields.iter().rposition(|f| {
        f.ty.traversal().is_some()
            || record(f, records).is_some_and(|r| {
                r.fields.iter().any(|f| {
                    matches!(&f.ty, PropertyType::Named(t) if t == "Value")
                        || matches!(&f.ty, PropertyType::Optional(_))
                })
            })
    })
}

fn omitted(layout: &Layout, records: &[RecordDef], i: usize) -> bool {
    last_group(layout, records) == Some(i)
        && matches!(
            layout.fields[i].ty.traversal(),
            Some("value_list" | "block_call")
        )
}

fn field_type(layout: &Layout, records: &[RecordDef], i: usize) -> Option<String> {
    let f = &layout.fields[i];
    if omitted(layout, records, i) {
        return f.ty.named("BlockCall").then(|| "crate::Block".into());
    }
    if f.ty.named("Int") || f.ty.named("Float") {
        Some("storage::ScalarBits".into())
    } else if f.ty.named("VectorConst") {
        Some("storage::VectorBits".into())
    } else if f.ty.named("u64") {
        Some("[u8; 8]".into())
    } else {
        stored_type(f, records)
    }
}

fn inline(layout: &Layout, records: &[RecordDef]) -> bool {
    let mut bytes = 1; // Enum tag. All remaining fields have at most u32 alignment.
    for (i, f) in layout.fields.iter().enumerate() {
        if let Some(r) = record(f, records) {
            let mut n = 0;
            let mut align = 1;
            for member in &r.fields {
                let s = match &member.ty {
                    PropertyType::Values(_) => {
                        unreachable!("nested fixed SSA arrays rejected by storage checking")
                    }
                    PropertyType::Named(t) if t == "Value" => continue,
                    PropertyType::Optional(_) => 1,
                    PropertyType::Named(t) => match size(t) {
                        Some(s) if s <= 4 => s,
                        _ => return false,
                    },
                };
                n += s;
                align = align.max(s);
            }
            bytes += n.next_multiple_of(align);
        } else {
            bytes += match f.ty.traversal() {
                Some("value" | "array") => 0,
                Some("value_list") => {
                    if omitted(layout, records, i) {
                        0
                    } else {
                        4
                    }
                }
                Some("block_call") => {
                    if omitted(layout, records, i) {
                        4
                    } else {
                        8
                    }
                }
                Some("jump_table") => return false,
                _ => match size(&f.ty.rust_type()) {
                    Some(s) => s,
                    None => return false,
                },
            };
        }
    }
    bytes <= 16
}

fn pattern(layout: &Layout, records: &[RecordDef], packed: bool, prefix: &str) -> String {
    construct(
        &format!("{prefix}::{}", layout.name),
        layout
            .fields
            .iter()
            .enumerate()
            .filter(|(i, f)| {
                if packed {
                    field_type(layout, records, *i).is_some()
                } else {
                    stored_type(f, records).is_some()
                }
            })
            .map(|(i, f)| (f.name.clone(), format!("_f{i}"))),
    )
}

pub(super) fn generate(layouts: &[Layout], records: &[RecordDef]) -> String {
    let hot: Vec<_> = layouts.iter().filter(|l| inline(l, records)).collect();
    let mut out =
        String::from("#[derive(Debug, Clone)] pub(crate) enum PackedFields {\nOutOfLine(u32),\n");
    for layout in &hot {
        writeln!(
            out,
            "{},",
            construct(
                &layout.name,
                layout
                    .fields
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(i, f)| field_type(layout, records, i).map(|t| (f.name.clone(), t))
                    )
            )
        )
        .unwrap();
    }
    out.push_str("}\n// Guard the actual target layout as schemas and Rust representations evolve.\nconst _: () = assert!(core::mem::size_of::<PackedFields>() <= 16);\n#[allow(unused_variables)] impl InstFields {\npub(crate) fn pack(self, pool: &mut storage::FieldPool) -> PackedFields { match self {\n");
    for layout in &hot {
        let fields = layout.fields.iter().enumerate().filter_map(|(i, f)| {
            field_type(layout, records, i)?;
            let value = if f.ty.named("Int") || f.ty.named("Float") {
                format!("storage::ScalarBits::new(_f{i}.into())")
            } else if f.ty.named("VectorConst") {
                format!("storage::VectorBits::new(_f{i})")
            } else if f.ty.named("u64") {
                format!("_f{i}.to_le_bytes()")
            } else if omitted(layout, records, i) {
                format!("_f{i}.block")
            } else {
                format!("_f{i}")
            };
            Some((f.name.clone(), value))
        });
        writeln!(
            out,
            "{} => {},",
            pattern(layout, records, false, "Self"),
            construct(&format!("PackedFields::{}", layout.name), fields)
        )
        .unwrap();
    }
    out.push_str("fields => PackedFields::OutOfLine(pool.insert(fields)),\n} } }\n#[allow(unused_variables)] impl PackedFields {\npub(crate) fn release(&self, pool: &mut storage::FieldPool) { if let Self::OutOfLine(id) = self { pool.remove(*id); } }\n");
    out.push_str("pub(crate) fn opcode(&self, pool: &storage::FieldPool) -> Opcode { match self {\nSelf::OutOfLine(id) => pool.get(*id).opcode(),\n");
    for layout in &hot {
        let opcode = match &layout.opcode {
            OpcodeSource::Fixed(op) => format!("Opcode::{op}"),
            OpcodeSource::Dynamic(i) => format!("*_f{i}"),
        };
        writeln!(
            out,
            "{} => {opcode},",
            pattern(layout, records, true, "Self")
        )
        .unwrap();
    }
    out.push_str("} }\npub(crate) fn map_functions(&mut self, pool: &mut storage::FieldPool, mut map: impl FnMut(crate::FuncId) -> crate::FuncId) { match self {\nSelf::OutOfLine(id) => pool.get_mut(*id).map_functions(map),\n");
    for layout in &hot {
        let funcs: Vec<_> = layout
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.ty.named("FuncId"))
            .collect();
        if funcs.is_empty() {
            continue;
        }
        writeln!(out, "{} => {{", pattern(layout, records, true, "Self")).unwrap();
        for (i, _) in funcs {
            writeln!(out, "*_f{i} = map(*_f{i});").unwrap();
        }
        out.push_str("},\n");
    }
    out.push_str("_ => {},\n} }\npub(crate) fn view<'a>(&'a self, values: &'a [Value], pool: &'a storage::FieldPool) -> InstructionView<'a> {\nlet mut reader = storage::OperandReader(values);\nlet view = match self {\nSelf::OutOfLine(id) => return pool.get(*id).view(values),\n");
    for layout in &hot {
        writeln!(out, "{} => {{", pattern(layout, records, true, "Self")).unwrap();
        for (i, f) in layout.fields.iter().enumerate() {
            let expr = if f.ty.named("Int") {
                format!("_f{i}.int()")
            } else if f.ty.named("Float") {
                format!("_f{i}.float()")
            } else if f.ty.named("VectorConst") {
                format!("_f{i}.value()")
            } else if f.ty.named("u64") {
                format!("u64::from_le_bytes(*_f{i})")
            } else {
                match f.ty.traversal() {
                    Some("value_list") if omitted(layout, records, i) => {
                        "reader.take(reader.0.len())".into()
                    }
                    Some("block_call") if omitted(layout, records, i) => {
                        format!("Successor {{ block: *_f{i}, args: reader.take(reader.0.len()) }}")
                    }
                    _ => read_field(f, records, &format!("_f{i}")),
                }
            };
            writeln!(out, "let _v{i} = {expr};").unwrap();
        }
        writeln!(
            out,
            "{} }},",
            construct(
                &format!("InstructionView::{}", layout.name),
                layout
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (f.name.clone(), format!("_v{i}")))
            )
        )
        .unwrap();
    }
    out.push_str("};\ndebug_assert!(reader.0.is_empty(), \"unconsumed operands\");\nview\n} }\nimpl InstructionView<'_> {\npub fn to_draft(&self) -> InstDraft { match self {\n");
    for layout in layouts {
        let args = layout
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| match f.ty.traversal() {
                Some("value_list") => format!("_field{i}"),
                Some("array") => format!("**_field{i}"),
                Some("jump_table") => format!("_field{i}.iter()"),
                _ => format!("*_field{i}"),
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "{} => InstDraft::{}({args}),",
            layout.pattern(),
            super::constructor_name(&layout.name)
        )
        .unwrap();
    }
    out.push_str("} } }\n");
    out
}
