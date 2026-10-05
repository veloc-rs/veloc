//! Compact persistent fields; public views expose logical types.
use super::generate::{construct, read_field, record, stored_type};
use super::{Access, Layout, OpcodeSource};
use crate::model::records::{Placement, PropertyType, RecordDef};
use std::fmt::Write;

pub(crate) fn construction(
    op: &crate::model::Op,
    format: &super::Format,
    opcode: &str,
    local: impl Fn(&str) -> String,
) -> crate::generate::construction::Write {
    use crate::generate::construction::Write;
    use crate::model::Binding;
    let args = format.fields.iter().map(|field| {
        if field.ty.named("Opcode") {
            return format!("crate::Opcode::{opcode}");
        }
        match &op.bindings()[&field.name] {
            Binding::Name(name) => {
                let value = local(name);
                if field.policy.references.is_edge() { format!("({value}).as_view()") } else { value }
            }
            Binding::Array(items) => {
                let items = items.iter().map(|item| {
                    let Binding::Name(name) = item else { unreachable!("checked array binding") };
                    local(name)
                }).collect::<Vec<_>>();
                format!("[{}]", items.join(", "))
            }
            Binding::Table { cases, default } => format!(
                "({}).iter().map(crate::SuccessorData::as_view).chain(core::iter::once(({}).as_view()))",
                local(cases), local(default)
            ),
        }
    }).collect();
    Write {
        callee: format!("writer.{}", super::constructor_name(&format.name)),
        args,
    }
}

// Unknown or large properties stay out of line. This is a storage policy, not
// a restriction on the logical schema. Byte arrays avoid padding for 64-bit data.
fn size(ty: &str) -> Option<usize> {
    Some(match ty {
        "Opcode" | "IntCC" | "FloatCC" | "bool" | "u8" => 1,
        "MemFlags" | "Intrinsic" => 2,
        "u32" | "i32" | "FuncId" | "SigId" => 4,
        "u64" => 8,
        _ => return None,
    })
}

fn last_group(layout: &Layout, records: &[RecordDef]) -> Option<usize> {
    layout.fields.iter().rposition(|f| {
        f.access().is_some()
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
            layout.fields[i].access(),
            Some(Access::Values | Access::Edge)
        )
}

fn field_type(layout: &Layout, records: &[RecordDef], i: usize) -> Option<String> {
    let f = &layout.fields[i];
    if omitted(layout, records, i) {
        return (f.policy.references.is_edge()).then(|| "crate::Block".into());
    }
    if f.ty.named("u64") {
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
                    PropertyType::Values(_)
                    | PropertyType::Array(_, _)
                    | PropertyType::Sequence(_) => {
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
            bytes += match f.access() {
                Some(Access::Value | Access::Array) => 0,
                Some(Access::Values) => {
                    if omitted(layout, records, i) {
                        0
                    } else {
                        4
                    }
                }
                Some(Access::Edge) => {
                    if omitted(layout, records, i) {
                        4
                    } else {
                        8
                    }
                }
                Some(Access::Edges) => return false,
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
            let expr = if f.ty.named("u64") {
                format!("u64::from_le_bytes(*_f{i})")
            } else {
                match f.access() {
                    Some(Access::Values) if omitted(layout, records, i) => {
                        "reader.take(reader.0.len())".into()
                    }
                    Some(Access::Edge) if omitted(layout, records, i) => {
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
    out.push_str("};\ndebug_assert!(reader.0.is_empty(), \"unconsumed operands\");\nview\n}\n");
    edge_access(&mut out, layouts, records, false);
    edge_access(&mut out, layouts, records, true);
    out.push_str("}\n");
    out
}

/// Locate successor arguments from the same storage layout used by the reader.
/// Edits report original offsets, so callers can compact operands in one pass.
fn edge_access(out: &mut String, layouts: &[Layout], records: &[RecordDef], mutable: bool) {
    let (method, borrow, argument, get) = if mutable {
        ("edit_edges", "&mut ", "&mut storage::Edge", "get_mut")
    } else {
        ("visit_edges", "&", "storage::Edge", "get")
    };
    writeln!(out, "pub(crate) fn {method}({borrow}self, pool: {borrow}FieldPool, operand_len: u32, mut visit: impl FnMut({argument}, core::ops::Range<u32>)) {{ match self {{").unwrap();
    for layout in layouts.iter().filter(|layout| {
        layout
            .fields
            .iter()
            .any(|f| matches!(f.access(), Some(Access::Edge | Access::Edges)))
    }) {
        let hot = inline(layout, records);
        if hot {
            writeln!(out, "{} => {{", pattern(layout, records, true, "Self")).unwrap();
        } else {
            writeln!(out, "Self::{}(id) => {{", layout.name).unwrap();
            let fields = layout
                .fields
                .iter()
                .enumerate()
                .filter(|(_, f)| stored_type(f, records).is_some())
                .map(|(i, f)| (f.name.clone(), format!("_f{i}")));
            writeln!(
                out,
                "let {} = pool.{get}(*id);",
                construct(&format!("{}Fields", layout.name), fields)
            )
            .unwrap();
        }
        out.push_str("let mut offset = 0u32;\n");
        for (i, f) in layout.fields.iter().enumerate() {
            match f.access() {
                Some(Access::Value) => out.push_str("offset += 1;\n"),
                Some(Access::Array) => {
                    let super::FieldType::Values(n) = f.ty else {
                        unreachable!()
                    };
                    writeln!(out, "offset += {n};").unwrap();
                }
                Some(Access::Values) if hot && omitted(layout, records, i) => {
                    out.push_str("offset = operand_len;\n");
                }
                Some(Access::Values) => writeln!(out, "offset += *_f{i};").unwrap(),
                Some(Access::Edges) => {
                    writeln!(out, "_f{i}.{method}(&mut offset, &mut visit);").unwrap();
                }
                Some(Access::Edge) => {
                    out.push_str("let start = offset;\n");
                    if hot && omitted(layout, records, i) {
                        let mutability = if mutable { "mut " } else { "" };
                        writeln!(out, "let {mutability}edge = storage::Edge {{ block: *_f{i}, len: operand_len - offset }}; offset = operand_len;").unwrap();
                        if mutable {
                            writeln!(out, "visit(&mut edge, start..offset); *_f{i} = edge.block;")
                                .unwrap();
                        } else {
                            out.push_str("visit(edge, start..offset);\n");
                        }
                    } else {
                        writeln!(out, "offset += _f{i}.len;").unwrap();
                        let edge = if mutable {
                            format!("_f{i}")
                        } else {
                            format!("*_f{i}")
                        };
                        writeln!(out, "visit({edge}, start..offset);").unwrap();
                    }
                }
                None => {
                    if let Some(record) = record(f, records) {
                        for member in &record.fields {
                            match &member.ty {
                                PropertyType::Optional(_) => {
                                    writeln!(out, "offset += u32::from(_f{i}.{});", member.name)
                                        .unwrap()
                                }
                                PropertyType::Named(_) if member.policy.references.is_operand() => {
                                    out.push_str("offset += 1;\n")
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        out.push_str("debug_assert_eq!(offset, operand_len);\n},\n");
    }
    out.push_str("_ => {} } }\n");
}

/// Normalize packed field adapters into the common logical access plan.
pub(crate) fn inputs(
    op: &crate::model::Op,
    format: &super::Format,
) -> crate::model::access::Inputs {
    use crate::model::{Binding, access::Access};
    let mut inputs = crate::model::access::Inputs::new();
    for field in &format.fields {
        if field.ty.named("Opcode") {
            continue;
        }
        let value = Access::Field(field.name.clone());
        match &op.bindings()[&field.name] {
            Binding::Name(name) => {
                inputs.insert(name.clone(), value);
            }
            Binding::Array(args) => {
                for (index, arg) in args.iter().enumerate() {
                    let Binding::Name(name) = arg else {
                        unreachable!("checked array binding")
                    };
                    inputs.insert(name.clone(), Access::Index(Box::new(value.clone()), index));
                }
            }
            Binding::Table { cases, default } => {
                inputs.insert(cases.clone(), Access::SplitLast(Box::new(value.clone()), 1));
                inputs.insert(default.clone(), Access::SplitLast(Box::new(value), 0));
            }
        }
    }
    inputs
}
