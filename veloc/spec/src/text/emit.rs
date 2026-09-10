//! Typed Rust emission. No operation names or text-codec registry live here.
use std::fmt::Write;

use super::schema::{Atom, AtomKind, CallSignature, Item, Mode, Schema};
use crate::model::{Op, ParamKind};
use crate::records::RecordDef;
use crate::storage::{FieldType, Format};

fn local(op: &Op, path: &str) -> String {
    let mut parts = path.split('.');
    let root = parts.next().unwrap();
    let index = op
        .params
        .iter()
        .position(|p| p.name == root)
        .expect("checked text parameter");
    std::iter::once(format!("_p{index}"))
        .chain(parts.map(str::to_owned))
        .collect::<Vec<_>>()
        .join(".")
}

fn leaf(op: &Op, path: &str) -> String {
    local(op, path).replace('.', "_")
}

pub(super) fn parse(
    op: &Op,
    format: &Format,
    schema: &Schema,
    records: &[RecordDef],
    arity: Option<usize>,
) -> String {
    let mut out = String::new();
    if let Some(path) = &schema.flags {
        writeln!(out, "let {} = flags;", leaf(op, path)).unwrap();
    } else {
        out.push_str("if flags != crate::MemFlags::empty() { return Err(input.error(\"memory flags are not supported by this operation\")); }\n");
    }
    let variadic = match schema.args.as_slice() {
        [Item::Atom(atom)] if matches!(atom.kind, AtomKind::Values) => Some(atom),
        _ => None,
    };
    if let Some(atom) = variadic {
        if let Some(count) = arity {
            let name = leaf(op, &atom.path);
            for index in 0..count {
                if index != 0 {
                    out.push_str("input.expect(Kind::Comma)?;\n");
                }
                writeln!(out, "let {name}_{index} = self.value(input)?;").unwrap();
            }
            let values = (0..count)
                .map(|i| format!("{name}_{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "let {name}: [crate::Value; {count}] = [{values}];").unwrap();
        } else {
            writeln!(out, "let {} = self.values(input)?;", leaf(op, &atom.path)).unwrap();
        }
    } else {
        for (index, item) in schema.args.iter().enumerate() {
            if index != 0 {
                out.push_str("input.expect(Kind::Comma)?;\n");
            }
            parse_item(&mut out, op, item);
        }
    }
    if !schema.named.is_empty() {
        // Each field has a statically typed slot: no runtime field registry,
        // string pairs, or repeated searches through a temporary field list.
        for named in &schema.named {
            writeln!(out, "let mut {} = None;", leaf(op, &named.atom.path)).unwrap();
        }
        out.push_str("if !input.at_end() {\n");
        if let (Some(atom), None) = (variadic, arity) {
            writeln!(
                out,
                "if !{}.is_empty() {{ input.expect(Kind::Comma)?; }}",
                leaf(op, &atom.path)
            )
            .unwrap();
        } else if arity.unwrap_or(schema.args.len()) != 0 {
            out.push_str("input.expect(Kind::Comma)?;\n");
        }
        out.push_str(
            "loop {\nlet _key_location = input.location();\nlet _key = input.word()?;\ninput.expect(Kind::Equal)?;\nmatch _key {\n",
        );
        for named in &schema.named {
            let name = leaf(op, &named.atom.path);
            writeln!(out, "{:?} => {{ if {name}.is_some() {{ return Err(_key_location.error(format!(\"duplicate `{{_key}}` field\"))); }} {name} = Some({}); }},", named.key, parse_atom(&named.atom)).unwrap();
        }
        out.push_str("_ => return Err(_key_location.error(format!(\"unknown named field `{_key}`\"))),\n}\nif !input.eat(Kind::Comma) { break; }\n} }\n");
        for named in &schema.named {
            let name = leaf(op, &named.atom.path);
            match named.mode {
                Mode::Required => writeln!(
                    out,
                    "let {name} = {name}.ok_or_else(|| input.error({:?}))?;",
                    format!("missing `{}=` field", named.key)
                )
                .unwrap(),
                Mode::Optional => {}
                Mode::Default(default) => {
                    writeln!(out, "let {name} = {name}.unwrap_or({default});").unwrap()
                }
            }
        }
    }
    for (path, default) in &schema.defaults {
        writeln!(out, "let {} = {};", leaf(op, path), default.rust()).unwrap();
    }
    for param in &op.params {
        if let ParamKind::Property(ty) = &param.kind
            && let Some(record) = records.iter().find(|r| r.name == *ty)
        {
            let fields = record
                .fields
                .iter()
                .map(|f| {
                    format!(
                        "{}: {}",
                        f.name,
                        leaf(op, &format!("{}.{}", param.name, f.name))
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                out,
                "let {} = crate::inst::{} {{ {fields} }};",
                local(op, &param.name),
                record.name
            )
            .unwrap();
        }
    }
    writeln!(
        out,
        "Ok({})",
        crate::packing::constructor(op, format, "self.func.dfg", |name| {
            let value = local(op, name);
            if op
                .params
                .iter()
                .any(|p| p.name == name && p.kind == ParamKind::Values)
            {
                format!("&{value}")
            } else {
                value
            }
        })
    )
    .unwrap();
    out
}

fn parse_item(out: &mut String, op: &Op, item: &Item) {
    match item {
        Item::Atom(atom) => {
            writeln!(out, "let {} = {};", leaf(op, &atom.path), parse_atom(atom)).unwrap();
        }
        Item::Space(lhs, rhs) => {
            parse_item(out, op, lhs);
            parse_item(out, op, rhs);
        }
        Item::Invoke {
            callee,
            args,
            signature,
        } => {
            writeln!(
                out,
                "let {} = {};",
                leaf(op, &callee.path),
                parse_atom_with(callee, &callee_codec(callee))
            )
            .unwrap();
            out.push_str("input.expect(Kind::LParen)?;\n");
            writeln!(out, "let {} = {};", leaf(op, &args.path), parse_atom(args)).unwrap();
            out.push_str("input.expect(Kind::RParen)?;\n");
            if !matches!(signature, CallSignature::Value) {
                out.push_str("input.expect(Kind::Colon).map_err(|e| e.context(\"signature must follow `:`\"))?;\n");
            }
            match signature {
                CallSignature::Value => {}
                CallSignature::Field(sig) => {
                    writeln!(out, "let {} = {};", leaf(op, &sig.path), parse_atom(sig)).unwrap();
                    if is_function(callee) {
                        writeln!(
                            out,
                            "let {name} = self.function_reference({name}, {})?;",
                            leaf(op, &sig.path),
                            name = leaf(op, &callee.path)
                        )
                        .unwrap();
                    }
                }
                CallSignature::Function => {
                    writeln!(
                        out,
                        "let _signature = self.signature(input)?;\nlet {name} = self.function_reference({name}, _signature)?;",
                        name = leaf(op, &callee.path)
                    )
                    .unwrap();
                }
            }
        }
    }
}

// One codec identity drives both directions; all codecs share the same API.
fn codec(kind: &AtomKind) -> String {
    match kind {
        AtomKind::Value | AtomKind::OptionalValue => "crate::Value".into(),
        AtomKind::Values => "super::atom::Values".into(),
        AtomKind::Successor => "crate::BlockCall".into(),
        AtomKind::Successors => "super::atom::Successors".into(),
        AtomKind::Integer => "super::atom::IntegerBits".into(),
        AtomKind::Float => "super::atom::FloatBits".into(),
        AtomKind::Bytes => "super::atom::Bytes".into(),
        AtomKind::Scalar(ty) => match ty.as_str() {
            "u8" | "u32" | "u64" | "i32" => format!("super::atom::Decimal<{ty}>"),
            "bool" => "bool".into(),
            _ => format!("crate::{ty}"),
        },
    }
}

fn is_function(atom: &Atom) -> bool {
    matches!(&atom.kind, AtomKind::Scalar(ty) if ty == "FuncId")
}

fn callee_codec(atom: &Atom) -> String {
    if is_function(atom) {
        "super::atom::FunctionName".into()
    } else {
        codec(&atom.kind)
    }
}

fn parse_atom(atom: &Atom) -> String {
    parse_atom_with(atom, &codec(&atom.kind))
}

fn parse_atom_with(atom: &Atom, codec: &str) -> String {
    format!(
        "<{} as super::atom::AtomCodec>::parse(self, input).map_err(|e| e.context({:?}))?",
        codec,
        format!("operand `{}`", atom.path)
    )
}

pub(super) fn print(
    canonical: &Op,
    op: &Op,
    format: &Format,
    schema: &Schema,
    arity: Option<usize>,
) -> String {
    let mut out = String::new();
    let fields = format
        .fields
        .iter()
        .enumerate()
        .map(|(i, f)| format!("{}: _s{i}", f.name))
        .collect::<Vec<_>>()
        .join(", ");
    if fields.is_empty() {
        writeln!(out, "crate::InstructionView::{} => {{", format.name).unwrap();
    } else {
        writeln!(
            out,
            "crate::InstructionView::{} {{ {fields} }} => {{",
            format.name
        )
        .unwrap();
    }
    for (index, field) in format.fields.iter().enumerate() {
        let expected = match &field.ty {
            FieldType::Named(ty) if ty == "ValueList" => arity,
            _ => None,
        };
        if let Some(n) = expected {
            writeln!(
                out,
                "if _s{index}.len() != {n} {{ return Err(core::fmt::Error); }}"
            )
            .unwrap();
        }
    }
    for (name, expr) in crate::packing::projections(
        op,
        format,
        "self.dfg",
        |name| {
            let index = format.fields.iter().position(|f| f.name == name).unwrap();
            format!("*_s{index}")
        },
        |value| format!("{value}.ok_or(core::fmt::Error)?"),
    ) {
        writeln!(out, "let {} = {expr};", local(op, &name)).unwrap();
    }
    for (path, default) in &schema.defaults {
        writeln!(
            out,
            "if {} != {} {{ return Err(core::fmt::Error); }}",
            local(op, path),
            default.rust()
        )
        .unwrap();
    }
    let flags = schema
        .flags
        .as_ref()
        .map(|p| local(op, p))
        .unwrap_or_else(|| "crate::MemFlags::empty()".into());
    writeln!(out, "self.fmt_head(f, {:?}, {flags})?;", canonical.mnemonic).unwrap();
    if !schema.args.is_empty() || !schema.named.is_empty() {
        out.push_str("let mut _separator = \" \";\n");
    }
    for item in &schema.args {
        let conditional = if let Item::Atom(Atom {
            path,
            kind: AtomKind::Values,
        }) = item
        {
            writeln!(out, "if !{}.is_empty() {{", local(op, path)).unwrap();
            true
        } else {
            false
        };
        out.push_str("f.write_str(_separator)?; _separator = \", \";\n");
        print_item(&mut out, op, item);
        if conditional {
            out.push_str("}\n");
        }
    }
    for named in &schema.named {
        let value = local(op, &named.atom.path);
        match named.mode {
            Mode::Default(n) => {
                writeln!(out, "if {value} != {n} {{").unwrap();
            }
            Mode::Optional => {
                writeln!(out, "if let Some(_value) = {value} {{").unwrap();
            }
            Mode::Required => {}
        }
        out.push_str("f.write_str(_separator)?; _separator = \", \";\n");
        writeln!(out, "f.write_str({:?})?;", format!("{}=", named.key)).unwrap();
        if matches!(named.mode, Mode::Optional) {
            out.push_str(
                "<crate::Value as super::atom::AtomCodec>::print(self, f, &_value, ty)?;\n",
            );
        } else {
            print_atom(&mut out, &named.atom, &value);
        }
        if !matches!(named.mode, Mode::Required) {
            out.push_str("}\n");
        }
    }
    out.push_str("Ok(())\n},\n");
    out
}

fn print_item(out: &mut String, op: &Op, item: &Item) {
    match item {
        Item::Atom(atom) => print_atom(out, atom, &local(op, &atom.path)),
        Item::Space(lhs, rhs) => {
            print_item(out, op, lhs);
            out.push_str("f.write_char(' ')?;\n");
            print_item(out, op, rhs);
        }
        Item::Invoke {
            callee,
            args,
            signature,
        } => {
            print_atom_with(out, callee, &local(op, &callee.path), &callee_codec(callee));
            out.push_str("f.write_char('(')?;\n");
            print_atom(out, args, &local(op, &args.path));
            out.push_str("f.write_char(')')?;\n");
            if !matches!(signature, CallSignature::Value) {
                out.push_str("f.write_str(\" : \")?;\n");
            }
            match signature {
                CallSignature::Value => {}
                CallSignature::Field(sig) => print_atom(out, sig, &local(op, &sig.path)),
                CallSignature::Function => {
                    writeln!(
                        out,
                        "self.fmt_function_signature(f, {})?;",
                        local(op, &callee.path)
                    )
                    .unwrap();
                }
            }
        }
    }
}

fn print_atom(out: &mut String, atom: &Atom, value: &str) {
    print_atom_with(out, atom, value, &codec(&atom.kind));
}

fn print_atom_with(out: &mut String, atom: &Atom, value: &str, codec: &str) {
    // Pool projections and variadic groups already yield borrowed slices.
    let value = if matches!(atom.kind, AtomKind::Values | AtomKind::Bytes) {
        value.to_owned()
    } else {
        format!("&{value}")
    };
    writeln!(
        out,
        "<{} as super::atom::AtomCodec>::print(self, f, {value}, ty)?;",
        codec
    )
    .unwrap();
}
