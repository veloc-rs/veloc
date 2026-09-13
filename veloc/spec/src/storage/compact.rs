//! Compact persistent fields; public views expose logical types.
use super::generate::{construct, read_field, record, stored_type};
use super::{Layout, OpcodeSource};
use crate::model::records::{Placement, PropertyType, RecordDef};
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
        f.traversal().is_some()
            || record(f, records).is_some_and(|r| {
                r.fields.iter().any(|f| {
                    matches!(&f.ty, PropertyType::Named(_) if f.policy.references.is_operand())
                        || matches!(&f.ty, PropertyType::Optional(_))
                })
            })
    })
}

fn omitted(layout: &Layout, records: &[RecordDef], i: usize) -> bool {
    last_group(layout, records) == Some(i)
        && matches!(
            layout.fields[i].traversal(),
            Some("value_list" | "block_call")
        )
}

fn field_type(layout: &Layout, records: &[RecordDef], i: usize) -> Option<String> {
    let f = &layout.fields[i];
    if omitted(layout, records, i) {
        return (f.policy.references.is_edge()).then(|| "crate::Block".into());
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

pub(super) fn inline(layout: &Layout, records: &[RecordDef]) -> bool {
    let mut bytes = 1; // Enum tag. All remaining fields have at most u32 alignment.
    for (i, f) in layout.fields.iter().enumerate() {
        if f.policy.storage == Placement::Pooled {
            return false;
        }
        if let Some(r) = record(f, records) {
            let mut n = 0;
            let mut align = 1;
            for member in &r.fields {
                if member.policy.storage == Placement::Pooled {
                    return false;
                }
                let s = match &member.ty {
                    PropertyType::Values(_) | PropertyType::Array(_, _) => {
                        unreachable!("nested fixed SSA arrays rejected by storage checking")
                    }
                    PropertyType::Named(_) if member.policy.references.is_operand() => continue,
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
            bytes += match f.traversal() {
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
                _ => match size(&f.ty.schema_type()) {
                    Some(s) => s,
                    None => return false,
                },
            };
        }
    }
    bytes <= 16
}

/// Encode typed constructor locals directly, without an owning draft enum.
pub(super) fn encode(
    layout: &Layout,
    records: &[RecordDef],
    local: impl Fn(usize) -> String,
) -> String {
    let hot = inline(layout, records);
    let fields = layout.fields.iter().enumerate().filter_map(|(i, f)| {
        if hot {
            field_type(layout, records, i)?;
        } else {
            stored_type(f, records)?;
        }
        let value = local(i);
        let value = if !hot {
            value
        } else if f.ty.named("Int") || f.ty.named("Float") {
            format!("storage::ScalarBits::new({value}.into())")
        } else if f.ty.named("VectorConst") {
            format!("storage::VectorBits::new({value})")
        } else if f.ty.named("u64") {
            format!("{value}.to_le_bytes()")
        } else if omitted(layout, records, i) {
            format!("{value}.block")
        } else {
            value
        };
        Some((f.name.clone(), value))
    });
    if hot {
        construct(&format!("InstFields::{}", layout.name), fields)
    } else {
        format!(
            "InstFields::{}(self.dfg.fields.push({}))",
            layout.name,
            construct(&format!("{}Fields", layout.name), fields)
        )
    }
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
    let cold: Vec<_> = layouts.iter().filter(|l| !inline(l, records)).collect();
    let mut out = String::new();
    for layout in &cold {
        writeln!(
            out,
            "#[derive(Debug, Clone)] pub(crate) struct {}Fields {{",
            layout.name
        )
        .unwrap();
        for f in &layout.fields {
            if let Some(ty) = stored_type(f, records) {
                writeln!(out, "{}: {ty},", f.name).unwrap();
            }
        }
        out.push_str("}\n");
    }
    out.push_str("#[derive(Debug, Clone, Default)] pub(crate) struct PayloadPool {\n");
    for layout in &cold {
        writeln!(
            out,
            "{}: storage::Pool<{}Fields>,",
            super::constructor_name(&layout.name),
            layout.name
        )
        .unwrap();
    }
    out.push_str("}\n");
    for layout in &cold {
        let name = &layout.name;
        let field = super::constructor_name(name);
        writeln!(
            out,
            "impl storage::Pooled for {name}Fields {{
            fn pool(pools: &FieldPool) -> &storage::Pool<Self> {{ &pools.payloads.{field} }}
            fn pool_mut(pools: &mut FieldPool) -> &mut storage::Pool<Self> {{ &mut pools.payloads.{field} }}
        }}"
        )
        .unwrap();
    }
    out.push_str("#[derive(Debug, Clone)] pub(crate) enum InstFields {\n");
    for layout in &cold {
        writeln!(out, "{}(storage::Id<{}Fields>),", layout.name, layout.name).unwrap();
    }
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
    out.push_str("}\nconst _: () = assert!(core::mem::size_of::<InstFields>() <= 16);\n#[allow(unused_variables)] impl InstFields {\n");
    out.push_str("pub(crate) fn clone_in(&self, pool: &mut FieldPool) -> Self { match self {\n");
    for layout in &cold {
        writeln!(out, "Self::{0}(id) => {{ let fields = pool.get(*id).clone(); Self::{0}(pool.push(fields)) }},", layout.name).unwrap();
    }
    out.push_str("_ => self.clone(), } }\n");
    // A schema can have zero, one, or many pooled variants.
    out.push_str("#[allow(clippy::single_match)] pub(crate) fn release(&self, pool: &mut FieldPool) { match self {\n");
    for layout in &cold {
        writeln!(out, "Self::{}(id) => pool.remove(*id),", layout.name).unwrap();
    }
    out.push_str("_ => {} } }\n");
    out.push_str("pub(crate) fn opcode(&self, pool: &FieldPool) -> Opcode { match self {\n");
    for layout in &cold {
        let opcode = match &layout.opcode {
            OpcodeSource::Fixed(op) => format!("Opcode::{op}"),
            OpcodeSource::Dynamic(i) => format!("pool.get(*id).{}", layout.fields[*i].name),
        };
        writeln!(out, "Self::{}(id) => {opcode},", layout.name).unwrap();
    }
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
    out.push_str("} }\npub(crate) fn map_functions(&mut self, pool: &mut FieldPool, mut map: impl FnMut(crate::FuncId) -> crate::FuncId) { match self {\n");
    for layout in &cold {
        writeln!(out, "Self::{}(id) => {{", layout.name).unwrap();
        for f in &layout.fields {
            if f.ty.named("FuncId") {
                writeln!(
                    out,
                    "let fields = pool.get_mut(*id); fields.{0} = map(fields.{0});",
                    f.name
                )
                .unwrap();
            }
        }
        out.push_str("},\n");
    }
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
    out.push_str("_ => {},\n} }\npub(crate) fn view<'a>(&'a self, values: &'a [Value], pool: &'a FieldPool) -> InstView<'a> {\nlet mut reader = storage::OperandReader(values);\nlet view = match self {\n");
    for layout in &cold {
        writeln!(out, "Self::{}(id) => {{", layout.name).unwrap();
        let fields = layout
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| stored_type(f, records).is_some())
            .map(|(i, f)| (f.name.clone(), format!("_f{i}")));
        writeln!(
            out,
            "let {} = pool.get(*id);",
            construct(&format!("{}Fields", layout.name), fields)
        )
        .unwrap();
        for (i, f) in layout.fields.iter().enumerate() {
            writeln!(
                out,
                "let _v{i} = {};",
                read_field(f, records, &format!("_f{i}"))
            )
            .unwrap();
        }
        writeln!(
            out,
            "{} }},",
            construct(
                &format!("InstView::{}", layout.name),
                layout
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (f.name.clone(), format!("_v{i}")))
            )
        )
        .unwrap();
    }
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
                match f.traversal() {
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
                &format!("InstView::{}", layout.name),
                layout
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (f.name.clone(), format!("_v{i}")))
            )
        )
        .unwrap();
    }
    out.push_str("};\ndebug_assert!(reader.0.is_empty(), \"unconsumed operands\");\nview\n} }\n");
    out
}
