//! Generate construction data, zero-allocation views and SSA-free storage.
use super::{Field, FieldType, FormatSource, Layout, OpcodeSource, value_only};
use crate::model::records::{PropertyType, RecordDef};
use std::fmt::Write;

pub(super) fn record<'a>(field: &Field, records: &'a [RecordDef]) -> Option<&'a RecordDef> {
    records.iter().find(|r| field.ty.named(&r.name))
}

pub(super) fn stored_type(field: &Field, records: &[RecordDef]) -> Option<String> {
    if let Some(record) = record(field, records) {
        return Some(format!("{}Fields", record.name));
    }
    match field.traversal() {
        Some("value" | "array") => None,
        Some("value_list") => Some("u32".into()),
        Some("block_call") => Some("storage::Edge".into()),
        Some("jump_table") => Some("storage::Edges".into()),
        _ => Some(field.rust.clone()),
    }
}

fn view_type(field: &Field) -> String {
    if let FieldType::Values(n) = field.ty {
        return format!("&'a [Value; {n}]");
    }
    match field.traversal() {
        Some("value_list") => "&'a [Value]".into(),
        Some("block_call") => "Successor<'a>".into(),
        Some("jump_table") => "Successors<'a>".into(),
        _ => field.rust.clone(),
    }
}

/// Decode one logical field from stored metadata and the shared operand reader.
pub(super) fn read_field(field: &Field, records: &[RecordDef], value: &str) -> String {
    if record(field, records).is_some() {
        return format!("{value}.view(&mut reader)");
    }
    match field.traversal() {
        Some("value") => "reader.value()".into(),
        Some("array") => {
            let FieldType::Values(n) = field.ty else {
                unreachable!()
            };
            format!("reader.take({n}).try_into().unwrap()")
        }
        Some("value_list") => format!("reader.take(*{value} as usize)"),
        Some("block_call") => format!("reader.edge(*{value})"),
        Some("jump_table") => format!("reader.edges({value})"),
        _ => format!("*{value}"),
    }
}

pub(super) fn construct(name: &str, fields: impl Iterator<Item = (String, String)>) -> String {
    let fields = fields
        .map(|(name, value)| {
            if name == value {
                name
            } else {
                format!("{name}: {value}")
            }
        })
        .collect::<Vec<_>>();
    if fields.is_empty() {
        name.into()
    } else {
        format!("{name} {{ {} }}", fields.join(", "))
    }
}

pub(super) fn instructions(layouts: &[Layout], records: &[RecordDef]) -> String {
    let mut out =
        String::from("// generated: construction data and borrowed views share one schema.\n");
    out.push_str("#[derive(Debug, Clone, Copy)] pub enum InstView<'a> {\n");
    for layout in layouts {
        if layout.fields.is_empty() {
            writeln!(out, "{},", layout.name).unwrap();
            continue;
        }
        writeln!(out, "{} {{", layout.name).unwrap();
        for field in &layout.fields {
            writeln!(out, "{}: {},", field.name, view_type(field)).unwrap();
        }
        out.push_str("},\n");
    }
    out.push_str("}\n");
    for record in records {
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy)] pub(crate) struct {}Fields {{",
            record.name
        )
        .unwrap();
        for f in &record.fields {
            let ty = match &f.ty {
                PropertyType::Named(_) if f.policy.references.is_operand() => continue,
                PropertyType::Optional(_) if f.policy.references.is_operand() => "bool",
                PropertyType::Named(_) => &f.rust,
                _ => unreachable!("checked record type"),
            };
            writeln!(out, "{}: {ty},", f.name).unwrap();
        }
        out.push_str("}\n");
        writeln!(out, "#[allow(unused_variables)] impl {} {{ fn store(self, values: &mut Arguments) -> {}Fields {{", record.name, record.name).unwrap();
        for f in &record.fields {
            match &f.ty {
                PropertyType::Named(_) if f.policy.references.is_operand() => {
                    writeln!(out, "values.push(self.{});", f.name).unwrap()
                }
                PropertyType::Optional(_) => {
                    writeln!(out, "values.extend(self.{});", f.name).unwrap()
                }
                _ => {}
            }
        }
        let fields = record.fields.iter().filter_map(|f| match &f.ty {
            PropertyType::Named(_) if f.policy.references.is_operand() => None,
            PropertyType::Optional(_) => {
                Some((f.name.clone(), format!("self.{}.is_some()", f.name)))
            }
            _ => Some((f.name.clone(), format!("self.{}", f.name))),
        });
        writeln!(
            out,
            "{} }} }}",
            construct(&format!("{}Fields", record.name), fields)
        )
        .unwrap();
        writeln!(out, "#[allow(unused_variables)] impl {}Fields {{ fn view(self, reader: &mut storage::OperandReader<'_>) -> {} {{", record.name, record.name).unwrap();
        let fields = record.fields.iter().map(|f| {
            let expr = match &f.ty {
                PropertyType::Named(_) if f.policy.references.is_operand() => {
                    "reader.value()".into()
                }
                PropertyType::Optional(_) => format!("self.{}.then(|| reader.value())", f.name),
                _ => format!("self.{}", f.name),
            };
            (f.name.clone(), expr)
        });
        writeln!(out, "{} }} }}", construct(&record.name, fields)).unwrap();
    }
    out.push_str("#[allow(unused_variables, unused_mut)] impl InstWriter<'_> {\n");
    for layout in layouts {
        // Keep internal locals disjoint from all definition-owned field names.
        let mut prefix = "_".to_owned();
        while layout.fields.iter().any(|f| f.name.starts_with(&prefix)) {
            prefix.push('_');
        }
        let values = format!("{prefix}operands");
        let params = layout
            .fields
            .iter()
            .map(|f| {
                let ty = match f.traversal() {
                    Some("value_list") => "&[Value]".into(),
                    Some("block_call") => "Successor<'_>".into(),
                    Some("jump_table") => "impl IntoIterator<Item = Successor<'a>>".into(),
                    _ => f.rust.clone(),
                };
                format!("{}: {ty}", f.name)
            })
            .collect::<Vec<_>>()
            .join(", ");
        let lifetime = if layout.fields.iter().any(|f| f.policy.references.is_edges()) {
            "<'a>"
        } else {
            ""
        };
        writeln!(out, "/// Construct this layout without validating its type contract.\npub fn {}{lifetime}(self, {params}) -> Inst {{\nlet mut {values} = Arguments::new();", super::constructor_name(&layout.name)).unwrap();
        for (i, f) in layout.fields.iter().enumerate() {
            let name = &f.name;
            let expr = if record(f, records).is_some() {
                format!("{name}.store(&mut {values})")
            } else {
                match f.traversal() {
                    Some("value") => {
                        writeln!(out, "{values}.push({name});").unwrap();
                        continue;
                    }
                    Some("array") => {
                        writeln!(out, "{values}.extend_from_slice(&{name});").unwrap();
                        continue;
                    }
                    Some("value_list") => {
                        writeln!(out, "{values}.extend_from_slice({name});").unwrap();
                        format!("u32::try_from({name}.len()).expect(\"too many operands\")")
                    }
                    Some("block_call") => format!("storage::store_edge({name}, &mut {values})"),
                    Some("jump_table") => format!("storage::Edges::store({name}, &mut {values})"),
                    _ => name.clone(),
                }
            };
            writeln!(out, "let {prefix}stored{i} = {expr};").unwrap();
        }
        writeln!(
            out,
            "let fields = {}; self.write(fields, &{values}) }}",
            super::compact::encode(layout, records, |i| format!("{prefix}stored{i}"))
        )
        .unwrap();
    }
    from_values(&mut out, layouts);
    out.push_str("}\n#[allow(unused_variables)] impl<'a> InstView<'a> {\npub fn opcode(&self) -> Opcode { match self {\n");
    for layout in layouts {
        let value = match &layout.opcode {
            OpcodeSource::Fixed(op) => format!("Opcode::{op}"),
            OpcodeSource::Dynamic(i) => format!("*_field{i}"),
        };
        writeln!(out, "{} => {value},", layout.pattern()).unwrap();
    }
    out.push_str("} }\npub fn matches_format(&self, format: OpFormat) -> bool { match self {\n");
    for layout in layouts {
        let condition = match &layout.format {
            FormatSource::Fixed(name) => format!("format == OpFormat::{name}"),
            FormatSource::Arity { field, formats } => format!(
                "matches!(format, {}) && format.fixed_value_arity() == Some(_field{field}.len())",
                formats
                    .iter()
                    .map(|f| format!("OpFormat::{f}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
        };
        writeln!(out, "{} => {condition},", layout.pattern()).unwrap();
    }
    out.push_str("} }\n");
    for (name, auxiliary) in [("visit_operands", true), ("visit_type_operands", false)] {
        let propagate = if auxiliary { "?" } else { "" };
        if auxiliary {
            writeln!(out, "pub fn {name}(&self, mut f: impl FnMut(Value)) {{ self.try_visit_operands::<core::convert::Infallible>(|value| {{ f(value); Ok(()) }}).unwrap_or_else(|never| match never {{}}); }}").unwrap();
            out.push_str("/// Visit all operands in storage order, stopping at the first error.\npub fn try_visit_operands<E>(&self, mut f: impl FnMut(Value) -> core::result::Result<(), E>) -> core::result::Result<(), E> { match self {\n");
        } else {
            writeln!(
                out,
                "pub fn {name}(&self, mut f: impl FnMut(Value)) {{ match self {{"
            )
            .unwrap();
        }
        for layout in layouts {
            writeln!(out, "{} => {{", layout.pattern()).unwrap();
            for (i, field) in layout.fields.iter().enumerate() {
                if let Some(record) = record(field, records) {
                    if !auxiliary {
                        continue;
                    }
                    for member in &record.fields {
                        match &member.ty {
                            PropertyType::Named(_) if member.policy.references.is_operand() => {
                                writeln!(out, "f(_field{i}.{})?;", member.name).unwrap()
                            }
                            PropertyType::Optional(_) => writeln!(
                                out,
                                "if let Some(value) = _field{i}.{} {{ f(value)?; }}",
                                member.name
                            )
                            .unwrap(),
                            _ => {}
                        }
                    }
                } else {
                    match field.traversal() {
                        Some("value") => writeln!(out, "f(*_field{i}){propagate};").unwrap(),
                        Some("array" | "value_list") => writeln!(out, "for &value in _field{i}.iter() {{ f(value){propagate}; }}").unwrap(),
                        Some("block_call") => writeln!(out, "for &value in _field{i}.args {{ f(value){propagate}; }}").unwrap(),
                        Some("jump_table") => writeln!(out, "for call in _field{i}.iter() {{ for &value in call.args {{ f(value){propagate}; }} }}").unwrap(),
                        _ => {}
                    }
                }
            }
            out.push_str("},\n");
        }
        out.push_str(if auxiliary { "} Ok(()) }\n" } else { "} }\n" });
    }
    out.push_str("pub fn memory_flags(&self) -> Option<MemFlags> { match self {\n");
    for layout in layouts {
        let expr = layout
            .fields
            .iter()
            .enumerate()
            .find_map(|(i, f)| {
                if f.ty.named("MemFlags") {
                    Some(format!("Some(*_field{i})"))
                } else {
                    record(f, records)
                        .and_then(|r| {
                            r.fields
                                .iter()
                                .find(|f| f.ty == PropertyType::Named("MemFlags".into()))
                        })
                        .map(|f| format!("Some(_field{i}.{})", f.name))
                }
            })
            .unwrap_or("None".into());
        writeln!(out, "{} => {expr},", layout.pattern()).unwrap();
    }
    out.push_str("} }\n}\n");
    out.push_str(&super::compact::generate(layouts, records));
    edit_successors(&mut out, layouts);
    out
}

fn from_values(out: &mut String, layouts: &[Layout]) {
    out.push_str("pub fn from_values(self, opcode: Opcode, values: &[Value]) -> Option<Inst> { match opcode.spec().format {\n");
    for layout in layouts
        .iter()
        .filter(|l| l.canonical && value_only(&l.fields))
    {
        let arity = layout.arity().unwrap();
        let mut index = 0;
        let fields = layout.fields.iter().map(|f| {
            let expr = match &f.ty {
                FieldType::Values(n) => {
                    let i = index;
                    index += n;
                    format!("values[{i}..{index}].try_into().unwrap()")
                }
                FieldType::Named(_) if f.policy.references.is_operand() => {
                    let i = index;
                    index += 1;
                    format!("values[{i}]")
                }
                _ => "opcode".into(),
            };
            (f.name.clone(), expr)
        });
        let check = if arity == 0 {
            "values.is_empty()".to_owned()
        } else {
            format!("values.len() == {arity}")
        };
        writeln!(
            out,
            "OpFormat::{} if {check} => Some(self.{}({})),",
            layout.name,
            super::constructor_name(&layout.name),
            fields
                .map(|(_, value)| value)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
    }
    out.push_str("_ => None, } }\n");
}

fn edit_successors(out: &mut String, layouts: &[Layout]) {
    out.push_str("impl crate::dfg::DataFlowGraph {\npub fn edit_successors(&mut self, inst: Inst, mut edit: impl FnMut(&mut SuccessorMut<'_>)) {\nmatch self.inst(inst) {\n");
    for layout in layouts {
        if !layout
            .fields
            .iter()
            .any(|f| matches!(f.traversal(), Some("block_call" | "jump_table")))
        {
            continue;
        }
        let pat = layout.pattern().replace("Self::", "InstView::");
        writeln!(out, "{pat} => {{").unwrap();
        for (i, f) in layout.fields.iter().enumerate() {
            let value = match f.traversal() {
                Some("array") => format!("*_field{i}"),
                Some("value_list") => format!("_field{i}.to_vec()"),
                Some("block_call") => {
                    format!("crate::BlockCall::new(_field{i}.block, _field{i}.args)")
                }
                Some("jump_table") => format!(
                    "_field{i}.iter().map(|s| crate::BlockCall::new(s.block, s.args)).collect::<alloc::vec::Vec<_>>()"
                ),
                _ => format!("_field{i}"),
            };
            let mutable = if matches!(f.traversal(), Some("block_call" | "jump_table")) {
                "mut "
            } else {
                ""
            };
            writeln!(out, "let {mutable}_arg{i} = {value};").unwrap();
        }
        for (i, f) in layout.fields.iter().enumerate() {
            match f.traversal() {
                Some("block_call") => {
                    writeln!(out, "SuccessorMut::edit_call(&mut _arg{i}, &mut edit);").unwrap()
                }
                Some("jump_table") => writeln!(
                    out,
                    "for call in &mut _arg{i} {{ SuccessorMut::edit_call(call, &mut edit); }}"
                )
                .unwrap(),
                _ => {}
            }
        }
        let args = layout
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| match f.traversal() {
                Some("value_list") => format!("&_arg{i}"),
                Some("block_call") => format!("_arg{i}.as_view()"),
                Some("jump_table") => format!("_arg{i}.iter().map(crate::BlockCall::as_view)"),
                _ => format!("_arg{i}"),
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "self.replace_inst(inst, |writer| writer.{}({args})); }},",
            super::constructor_name(&layout.name)
        )
        .unwrap();
    }
    out.push_str("_ => {}, } } }\n");
}
