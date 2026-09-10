//! Generate construction data, zero-allocation views and SSA-free storage.
use super::{Field, FieldType, FormatSource, Layout, OpcodeSource, value_only};
use crate::records::{PropertyType, RecordDef};
use std::fmt::Write;

fn record<'a>(field: &Field, records: &'a [RecordDef]) -> Option<&'a RecordDef> {
    records.iter().find(|r| field.ty.named(&r.name))
}

fn stored_type(field: &Field, records: &[RecordDef]) -> Option<String> {
    if let Some(record) = record(field, records) {
        return Some(format!("{}Fields", record.name));
    }
    match field.ty.traversal() {
        Some("value" | "array") => None,
        Some("value_list") => Some("u32".into()),
        Some("block_call") => Some("storage::Edge".into()),
        Some("jump_table") => Some("storage::Edges".into()),
        _ => Some(field.ty.rust_type()),
    }
}

fn view_type(field: &Field) -> String {
    if let FieldType::Values(n) = field.ty {
        return format!("&'a [Value; {n}]");
    }
    match field.ty.traversal() {
        Some("value_list") => "&'a [Value]".into(),
        Some("block_call") => "Successor<'a>".into(),
        Some("jump_table") => "Successors<'a>".into(),
        _ => field.ty.rust_type(),
    }
}

fn construct(name: &str, fields: impl Iterator<Item = (String, String)>) -> String {
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
        String::from("// @generated: construction data and borrowed views share one schema.\n");
    out.push_str("#[derive(Debug, Clone, Copy)] pub enum InstructionView<'a> {\n");
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
    out.push_str("#[derive(Debug, Clone)] pub(crate) enum InstFields {\n");
    for layout in layouts {
        let fields = layout
            .fields
            .iter()
            .filter_map(|f| stored_type(f, records).map(|ty| (f.name.clone(), ty)));
        writeln!(out, "{},", construct(&layout.name, fields)).unwrap();
    }
    out.push_str("}\n");
    out.push_str("impl InstFields { pub(crate) fn opcode(&self) -> Opcode { match self {\n");
    for layout in layouts {
        let fields = layout
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| stored_type(f, records).is_some())
            .map(|(i, f)| (f.name.clone(), format!("_field{i}")));
        let pat = construct(&format!("Self::{}", layout.name), fields);
        let opcode = match &layout.opcode {
            OpcodeSource::Fixed(op) => format!("Opcode::{op}"),
            OpcodeSource::Dynamic(i) => format!("*_field{i}"),
        };
        writeln!(out, "{pat} => {opcode},").unwrap();
    }
    out.push_str("} } }\n");
    out.push_str("impl InstFields { #[allow(clippy::single_match)] pub(crate) fn map_functions(&mut self, mut map: impl FnMut(crate::FuncId) -> crate::FuncId) { let _ = &mut map; match self {\n");
    for layout in layouts {
        let fields: Vec<_> = layout
            .fields
            .iter()
            .filter(|f| f.ty.named("FuncId"))
            .collect();
        if fields.is_empty() {
            continue;
        }
        let names = fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(out, "Self::{} {{ {names}, .. }} => {{", layout.name).unwrap();
        for field in fields {
            writeln!(out, "*{0} = map(*{0});", field.name).unwrap();
        }
        out.push_str("},\n");
    }
    out.push_str("_ => {},\n} } }\n");
    for record in records {
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy)] pub(crate) struct {}Fields {{",
            record.name
        )
        .unwrap();
        for f in &record.fields {
            let ty = match &f.ty {
                PropertyType::Named(ty) if ty == "Value" => continue,
                PropertyType::Optional(ty) if ty == "Value" => "bool",
                PropertyType::Named(ty) => ty,
                _ => unreachable!("checked record type"),
            };
            writeln!(out, "{}: {ty},", f.name).unwrap();
        }
        out.push_str("}\n");
        writeln!(out, "#[allow(unused_variables)] impl {} {{ fn store(self, values: &mut Arguments) -> {}Fields {{", record.name, record.name).unwrap();
        for f in &record.fields {
            match &f.ty {
                PropertyType::Named(ty) if ty == "Value" => {
                    writeln!(out, "values.push(self.{});", f.name).unwrap()
                }
                PropertyType::Optional(_) => {
                    writeln!(out, "values.extend(self.{});", f.name).unwrap()
                }
                _ => {}
            }
        }
        let fields = record.fields.iter().filter_map(|f| match &f.ty {
            PropertyType::Named(ty) if ty == "Value" => None,
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
                PropertyType::Named(ty) if ty == "Value" => "reader.value()".into(),
                PropertyType::Optional(_) => format!("self.{}.then(|| reader.value())", f.name),
                _ => format!("self.{}", f.name),
            };
            (f.name.clone(), expr)
        });
        writeln!(out, "{} }} }}", construct(&record.name, fields)).unwrap();
    }
    out.push_str("#[allow(unused_variables, unused_mut)] impl InstDraft {\n");
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
                let ty = match f.ty.traversal() {
                    Some("value_list") => "&[Value]".into(),
                    Some("block_call") => "Successor<'_>".into(),
                    Some("jump_table") => "impl IntoIterator<Item = Successor<'a>>".into(),
                    _ => f.ty.rust_type(),
                };
                format!("{}: {ty}", f.name)
            })
            .collect::<Vec<_>>()
            .join(", ");
        let lifetime = if layout.fields.iter().any(|f| f.ty.named("JumpTable")) {
            "<'a>"
        } else {
            ""
        };
        writeln!(out, "/// Construct this layout without validating its type contract.\npub fn {}{lifetime}({params}) -> Self {{\nlet mut {values} = Arguments::new();", super::constructor_name(&layout.name)).unwrap();
        let mut fields = Vec::new();
        for (i, f) in layout.fields.iter().enumerate() {
            let name = &f.name;
            let expr = if record(f, records).is_some() {
                format!("{name}.store(&mut {values})")
            } else {
                match f.ty.traversal() {
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
                        format!("{name}.len().try_into().expect(\"too many operands\")")
                    }
                    Some("block_call") => format!("storage::store_edge({name}, &mut {values})"),
                    Some("jump_table") => format!("storage::Edges::store({name}, &mut {values})"),
                    _ => name.clone(),
                }
            };
            writeln!(out, "let {prefix}stored{i} = {expr};").unwrap();
            fields.push((name.clone(), format!("{prefix}stored{i}")));
        }
        writeln!(
            out,
            "Self {{ fields: {}, operands: {values} }} }}",
            construct(&format!("InstFields::{}", layout.name), fields.into_iter())
        )
        .unwrap();
    }
    from_values(&mut out, layouts);
    successor_edit(&mut out, layouts, records);
    out.push_str("}\n#[allow(unused_variables)] impl InstFields {\npub(crate) fn view<'a>(&'a self, values: &'a [Value]) -> InstructionView<'a> {\nlet mut reader = storage::OperandReader(values);\nlet view = match self {\n");
    for layout in layouts {
        let bindings = layout
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| stored_type(f, records).is_some())
            .map(|(i, f)| (f.name.clone(), format!("_field{i}")));
        let pat = construct(&format!("Self::{}", layout.name), bindings);
        writeln!(out, "{pat} => {{").unwrap();
        let mut fields = Vec::new();
        for (i, f) in layout.fields.iter().enumerate() {
            let expr = if record(f, records).is_some() {
                format!("_field{i}.view(&mut reader)")
            } else {
                match f.ty.traversal() {
                    Some("value") => "reader.value()".into(),
                    Some("array") => {
                        let FieldType::Values(n) = f.ty else {
                            unreachable!()
                        };
                        format!("reader.take({n}).try_into().unwrap()")
                    }
                    Some("value_list") => format!("reader.take(*_field{i} as usize)"),
                    Some("block_call") => format!("reader.edge(*_field{i})"),
                    Some("jump_table") => format!("reader.edges(_field{i})"),
                    _ => format!("*_field{i}"),
                }
            };
            writeln!(out, "let _view{i} = {expr};").unwrap();
            fields.push((f.name.clone(), format!("_view{i}")));
        }
        writeln!(
            out,
            "{} }},",
            construct(
                &format!("InstructionView::{}", layout.name),
                fields.into_iter()
            )
        )
        .unwrap();
    }
    out.push_str("};\ndebug_assert!(reader.0.is_empty(), \"unconsumed operands\");\nview\n} }\n#[allow(unused_variables)] impl<'a> InstructionView<'a> {\npub fn opcode(&self) -> Opcode { match self {\n");
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
                            PropertyType::Named(ty) if ty == "Value" => {
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
                    match field.ty.traversal() {
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
    out
}

fn from_values(out: &mut String, layouts: &[Layout]) {
    out.push_str("pub fn from_values(opcode: Opcode, values: &[Value]) -> Option<Self> { match opcode.spec().format {\n");
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
                FieldType::Named(ty) if ty == "Value" => {
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
            "OpFormat::{} if {check} => Some(Self::{}({})),",
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

fn successor_edit(out: &mut String, layouts: &[Layout], records: &[RecordDef]) {
    out.push_str("/// Edit individual successor occurrences in storage order.\npub fn edit_successors(&mut self, mut f: impl FnMut(&mut SuccessorMut<'_>)) {\nlet mut offset = 0;\nmatch &mut self.fields {\n");
    for layout in layouts {
        let Some(last) = layout
            .fields
            .iter()
            .rposition(|f| matches!(f.ty.traversal(), Some("block_call" | "jump_table")))
        else {
            continue;
        };
        let fields = layout
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| stored_type(f, records).is_some())
            .map(|(i, f)| (f.name.clone(), format!("_field{i}")));
        writeln!(
            out,
            "{} => {{",
            construct(&format!("InstFields::{}", layout.name), fields)
        )
        .unwrap();
        for (i, field) in layout.fields[..=last].iter().enumerate() {
            if let Some(record) = record(field, records) {
                for member in &record.fields {
                    match &member.ty {
                        PropertyType::Named(ty) if ty == "Value" => out.push_str("offset += 1;\n"),
                        PropertyType::Optional(_) => {
                            writeln!(out, "offset += usize::from(_field{i}.{});", member.name)
                                .unwrap()
                        }
                        _ => {}
                    }
                }
            } else {
                match &field.ty {
                    FieldType::Values(n) => writeln!(out, "offset += {n};").unwrap(),
                    _ => match field.ty.traversal() {
                        Some("value") => out.push_str("offset += 1;\n"),
                        Some("value_list") => {
                            writeln!(out, "offset += *_field{i} as usize;").unwrap()
                        }
                        Some("block_call" | "jump_table") => writeln!(
                            out,
                            "_field{i}.edit(&mut self.operands, &mut offset, &mut f);"
                        )
                        .unwrap(),
                        _ => {}
                    },
                }
            }
        }
        out.push_str("},\n");
    }
    out.push_str("_ => {},\n}\n}\n");
}
