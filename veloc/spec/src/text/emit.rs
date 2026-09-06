//! Typed Rust emission. No operation names or text-codec registry live here.
use std::fmt::Write;

use super::schema::{Atom, AtomKind, Item, Mode, Schema};
use crate::model::{Op, ParamKind, TypeList};
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
        out.push_str("if flags != crate::MemFlags::empty() { return Err(ParseError(\"memory flags are not supported by this operation\".into())); }\n");
    }
    let variadic = matches!(
        schema.args.as_slice(),
        [Item::Atom(Atom {
            kind: AtomKind::Values,
            ..
        })]
    );
    if variadic && arity.is_none() && schema.named.is_empty() {
        let Item::Atom(atom) = &schema.args[0] else {
            unreachable!()
        };
        // An unbounded value list needs no split/rejoin or temporary text buffer.
        writeln!(
            out,
            "let {} = {};",
            leaf(op, &atom.path),
            parse_atom(atom, "text")
        )
        .unwrap();
    } else {
        if variadic && arity.is_none() {
            out.push_str("let (_core, _named) = split_core_and_named(text, None)?;\n");
        } else {
            let count = if variadic {
                arity.expect("alternate arity")
            } else {
                schema.args.len()
            };
            if schema.named.is_empty() {
                writeln!(out, "let _core = exact_fields(text, {count})?;").unwrap();
            } else {
                writeln!(
                    out,
                    "let (_core, _named) = split_core_and_named(text, Some({count}))?;"
                )
                .unwrap();
            }
        }
        if !schema.named.is_empty() {
            let keys = schema
                .named
                .iter()
                .map(|n| format!("{:?}", n.key))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "reject_unknown_named(&_named, &[{keys}])?;").unwrap();
        }
        if variadic {
            let Item::Atom(atom) = &schema.args[0] else {
                unreachable!()
            };
            writeln!(out, "let {} = _core.iter().map(|s| <crate::Value as super::atom::AtomCodec>::parse(self, s, ty)).collect::<core::result::Result<Vec<_>, _>>()?;", leaf(op, &atom.path)).unwrap();
        } else {
            for (index, item) in schema.args.iter().enumerate() {
                parse_item(
                    &mut out,
                    op,
                    item,
                    &format!("_core[{index}]"),
                    &format!("_t{index}"),
                );
            }
        }
        for named in &schema.named {
            let name = leaf(op, &named.atom.path);
            match named.mode {
                Mode::Required => {
                    let expr = parse_atom(
                        &named.atom,
                        &format!("named_value(&_named, {:?})?", named.key),
                    );
                    writeln!(out, "let {name} = {expr};").unwrap();
                }
                Mode::Optional => {
                    writeln!(
                        out,
                        "let {name} = _named.iter().find(|(key, _)| *key == {:?}).map(|(_, text)| <crate::Value as super::atom::AtomCodec>::parse(self, text, ty)).transpose()?;",
                        named.key
                    )
                    .unwrap();
                }
                Mode::Default(default) => {
                    let expr = parse_atom(&named.atom, "_text");
                    writeln!(out, "let {name} = match _named.iter().find(|(key, _)| *key == {:?}) {{ Some((_, _text)) => {expr}, None => {default} }};", named.key).unwrap();
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
        crate::packing::constructor(op, format, "self.func.dfg", |name| local(op, name))
    )
    .unwrap();
    out
}

fn parse_item(out: &mut String, op: &Op, item: &Item, text: &str, temp: &str) {
    match item {
        Item::Atom(atom) => {
            writeln!(
                out,
                "let {} = {};",
                leaf(op, &atom.path),
                parse_atom(atom, text)
            )
            .unwrap();
        }
        Item::Space(lhs, rhs) => {
            writeln!(out, "let ({temp}_lhs, {temp}_rhs) = split_space({text})?;").unwrap();
            parse_item(out, op, lhs, &format!("{temp}_lhs"), &format!("{temp}a"));
            parse_item(out, op, rhs, &format!("{temp}_rhs"), &format!("{temp}b"));
        }
        Item::Invoke {
            callee,
            args,
            signature,
        } => {
            writeln!(
                out,
                "let ({temp}_callee, {temp}_args, {temp}_tail) = parse_invocation({text})?;"
            )
            .unwrap();
            writeln!(
                out,
                "let {} = {};",
                leaf(op, &callee.path),
                parse_atom(callee, &format!("{temp}_callee"))
            )
            .unwrap();
            writeln!(
                out,
                "let {} = {};",
                leaf(op, &args.path),
                parse_atom(args, &format!("{temp}_args"))
            )
            .unwrap();
            if let Some(sig) = signature {
                writeln!(
                    out,
                    "let {} = {};",
                    leaf(op, &sig.path),
                    parse_atom(sig, &format!("signature_suffix({temp}_tail)?"))
                )
                .unwrap();
            } else {
                writeln!(out, "require_empty({temp}_tail)?;").unwrap();
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

fn parse_atom(atom: &Atom, text: &str) -> String {
    format!(
        "<{} as super::atom::AtomCodec>::parse(self, {text}, ty)?",
        codec(&atom.kind)
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
        writeln!(out, "crate::InstructionData::{} => {{", format.name).unwrap();
    } else {
        writeln!(
            out,
            "crate::InstructionData::{} {{ {fields} }} => {{",
            format.name
        )
        .unwrap();
    }
    for (index, field) in format.fields.iter().enumerate() {
        let expected = match &field.ty {
            FieldType::List(n) => Some(*n),
            FieldType::Named(ty) if ty == "ValueList" => arity,
            _ => None,
        };
        if let Some(n) = expected {
            writeln!(out, "if self.dfg.get_value_list(*_s{index}).len() != {n} {{ return Err(core::fmt::Error); }}").unwrap();
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
    let ty = if matches!(&canonical.signature.results, TypeList::Fixed(results) if results.is_empty())
    {
        "None"
    } else {
        "ty"
    };
    let flags = schema
        .flags
        .as_ref()
        .map(|p| local(op, p))
        .unwrap_or_else(|| "crate::MemFlags::empty()".into());
    writeln!(
        out,
        "self.fmt_head(f, {:?}, {ty}, {flags})?;",
        canonical.mnemonic
    )
    .unwrap();
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
            print_atom(out, callee, &local(op, &callee.path));
            out.push_str("f.write_char('(')?;\n");
            print_atom(out, args, &local(op, &args.path));
            out.push_str("f.write_char(')')?;\n");
            if let Some(sig) = signature {
                out.push_str("f.write_str(\" : \")?;\n");
                print_atom(out, sig, &local(op, &sig.path));
            }
        }
    }
}

fn print_atom(out: &mut String, atom: &Atom, value: &str) {
    // Pool projections and variadic groups already yield borrowed slices.
    let value = if matches!(
        atom.kind,
        AtomKind::Values | AtomKind::Successors | AtomKind::Bytes
    ) {
        value.to_owned()
    } else {
        format!("&{value}")
    };
    writeln!(
        out,
        "<{} as super::atom::AtomCodec>::print(self, f, {value}, ty)?;",
        codec(&atom.kind)
    )
    .unwrap();
}
