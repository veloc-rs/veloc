//! Checked text projections over logical parameters, not storage layouts.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::model::{Op, ParamKind, Pattern, TypeDef, TypeList};
use crate::records::{DefaultValue, PropertyType, RecordDef, numeric_default};
use crate::syntax::{Kind, Node};

#[derive(Debug)]
pub(super) struct Schema {
    pub args: Vec<Item>,
    pub named: Vec<Named>,
    pub flags: Option<String>,
    pub defaults: Vec<(String, DefaultValue)>,
}

#[derive(Debug)]
pub(super) enum Item {
    Atom(Atom),
    Space(Box<Item>, Box<Item>),
    Invoke {
        callee: Atom,
        args: Atom,
        signature: Option<Atom>,
    },
}

#[derive(Debug, Clone)]
pub(super) struct Atom {
    pub path: String,
    pub kind: AtomKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AtomKind {
    Value,
    Values,
    Successor,
    Successors,
    Scalar(String),
    OptionalValue,
    Integer,
    Float,
    Bytes,
}

#[derive(Debug)]
pub(super) struct Named {
    pub atom: Atom,
    pub key: String,
    pub mode: Mode,
}

#[derive(Debug)]
pub(super) enum Mode {
    Required,
    Optional,
    Default(u32),
}

struct Leaf {
    kind: AtomKind,
    default: Option<DefaultValue>,
}

struct Checker<'a> {
    source: &'a str,
    leaves: BTreeMap<String, Leaf>,
    used: BTreeSet<String>,
    float: bool,
}

pub(super) fn compile(op: &Op, records: &[RecordDef], source: &str) -> Result<Schema, Error> {
    let mut checker = Checker {
        source,
        leaves: BTreeMap::new(),
        used: BTreeSet::new(),
        float: false,
    };
    for param in &op.params {
        if let ParamKind::Property(ty) = &param.kind
            && let Some(record) = records.iter().find(|record| record.name == *ty)
        {
            for field in &record.fields {
                let kind = match &field.ty {
                    PropertyType::Named(ty) if ty == "Value" => AtomKind::Value,
                    PropertyType::Named(ty) => AtomKind::Scalar(ty.clone()),
                    PropertyType::Optional(ty) if ty == "Value" => AtomKind::OptionalValue,
                    _ => return Err(checker.error(op.offset, "unsupported optional property")),
                };
                checker.leaves.insert(
                    format!("{}.{}", param.name, field.name),
                    Leaf {
                        kind,
                        default: field.default.clone(),
                    },
                );
            }
            continue;
        }
        let kind = match &param.kind {
            ParamKind::Value => AtomKind::Value,
            ParamKind::Values => AtomKind::Values,
            ParamKind::Successor => AtomKind::Successor,
            ParamKind::Successors => AtomKind::Successors,
            ParamKind::Property(ty) => AtomKind::Scalar(ty.clone()),
        };
        checker.leaves.insert(
            param.name.clone(),
            Leaf {
                kind,
                default: None,
            },
        );
    }

    let mut schema = Schema {
        args: Vec::new(),
        named: Vec::new(),
        flags: None,
        defaults: Vec::new(),
    };
    if let Some(node) = &op.text {
        let Kind::Object(name, fields) = &node.kind else {
            return Err(checker.error(node.offset, "expected Text { args, named, flags }"));
        };
        if name != "Text" {
            return Err(checker.error(node.offset, "expected Text projection"));
        }
        for (key, value) in fields {
            match key.as_str() {
                "args" => {
                    for item in list(value, source)? {
                        schema.args.push(checker.item(item)?);
                    }
                }
                "named" => {
                    let mut keys = BTreeSet::new();
                    for item in list(value, source)? {
                        let named = checker.named(item)?;
                        if !keys.insert(named.key.clone()) {
                            return Err(checker.error(
                                item.offset,
                                format!("duplicate named key `{}`", named.key),
                            ));
                        }
                        schema.named.push(named);
                    }
                }
                "flags" => {
                    let path = path(value, source)?;
                    if checker.consume(path, value.offset)? != AtomKind::Scalar("MemFlags".into()) {
                        return Err(
                            checker.error(value.offset, "text flags must reference MemFlags")
                        );
                    }
                    schema.flags = Some(path.into());
                }
                _ => return Err(checker.error(value.offset, format!("unknown Text field `{key}`"))),
            }
        }
    } else {
        for param in &op.params {
            let node = Node {
                offset: op.offset,
                kind: Kind::Name(param.name.clone()),
            };
            schema.args.push(checker.item(&node)?);
        }
    }

    if schema.args.iter().any(|item| {
        matches!(
            item,
            Item::Atom(Atom {
                kind: AtomKind::Values,
                ..
            })
        )
    }) && schema.args.len() != 1
    {
        return Err(checker.error(
            op.offset,
            "top-level values must be the only positional item",
        ));
    }
    if checker.float && !float_result(&op.signature) {
        return Err(checker.error(
            op.offset,
            "float text atoms require a scalar float first result",
        ));
    }
    for (path, leaf) in checker.leaves {
        if checker.used.contains(&path) {
            continue;
        }
        match leaf.default {
            Some(default) => schema.defaults.push((path, default)),
            None => {
                return Err(Error::at(
                    source,
                    op.offset,
                    format!("text projection does not consume `{path}`"),
                ));
            }
        }
    }
    Ok(schema)
}

impl Checker<'_> {
    fn error(&self, offset: usize, message: impl Into<String>) -> Error {
        Error::at(self.source, offset, message)
    }

    fn consume(&mut self, path: &str, offset: usize) -> Result<AtomKind, Error> {
        let leaf = self.leaves.get(path).ok_or_else(|| {
            self.error(
                offset,
                format!("unknown text field `{path}`; records require a field path"),
            )
        })?;
        if !self.used.insert(path.into()) {
            return Err(self.error(
                offset,
                format!("text field `{path}` is consumed more than once"),
            ));
        }
        Ok(leaf.kind.clone())
    }

    fn atom(&mut self, node: &Node) -> Result<Atom, Error> {
        let (path, codec) = match &node.kind {
            Kind::Name(path) => (path.as_str(), None),
            Kind::Call(codec, args)
                if args.len() == 1 && matches!(codec.as_str(), "integer" | "float" | "bytes") =>
            {
                (path(&args[0], self.source)?, Some(codec.as_str()))
            }
            _ => return Err(self.error(node.offset, "expected a field path or typed text atom")),
        };
        let kind = self.consume(path, node.offset)?;
        let kind = match (codec, kind) {
            (Some("integer"), AtomKind::Scalar(ty)) if ty == "u64" => AtomKind::Integer,
            (Some("float"), AtomKind::Scalar(ty)) if ty == "u64" => {
                self.float = true;
                AtomKind::Float
            }
            (Some("bytes"), AtomKind::Scalar(ty)) if ty == "Bytes" => AtomKind::Bytes,
            (Some(codec), _) => {
                return Err(self.error(
                    node.offset,
                    format!("text codec `{codec}` is incompatible with `{path}`"),
                ));
            }
            (None, AtomKind::Scalar(ty)) if !simple_scalar(&ty) => {
                return Err(self.error(
                    node.offset,
                    format!("property `{path}` of type `{ty}` needs an explicit text projection"),
                ));
            }
            (None, kind) => kind,
        };
        Ok(Atom {
            path: path.into(),
            kind,
        })
    }

    fn item(&mut self, node: &Node) -> Result<Item, Error> {
        match &node.kind {
            Kind::Call(name, args) if name == "space" => {
                if args.len() != 2 {
                    return Err(self.error(node.offset, "space requires two single-token atoms"));
                }
                let lhs = self.atom(&args[0])?;
                let rhs = self.atom(&args[1])?;
                if !single_token(&lhs.kind) || !single_token(&rhs.kind) {
                    return Err(self.error(node.offset, "space requires two single-token atoms"));
                }
                Ok(Item::Space(
                    Box::new(Item::Atom(lhs)),
                    Box::new(Item::Atom(rhs)),
                ))
            }
            Kind::Call(name, args) if name == "invoke" => {
                if !(2..=3).contains(&args.len()) {
                    return Err(self.error(
                        node.offset,
                        "invoke requires callee, values and an optional signature",
                    ));
                }
                let callee = self.atom(&args[0])?;
                let values = self.atom(&args[1])?;
                let signature = args.get(2).map(|node| self.atom(node)).transpose()?;
                if !matches!(&callee.kind, AtomKind::Value)
                    && !matches!(&callee.kind, AtomKind::Scalar(ty) if matches!(ty.as_str(), "FuncId" | "Intrinsic"))
                {
                    return Err(self.error(
                        node.offset,
                        "invoke callee must be a value, FuncId or Intrinsic",
                    ));
                }
                if values.kind != AtomKind::Values {
                    return Err(self.error(node.offset, "invoke arguments must be values"));
                }
                if signature
                    .as_ref()
                    .is_some_and(|atom| atom.kind != AtomKind::Scalar("SigId".into()))
                {
                    return Err(self.error(node.offset, "invoke signature must be a SigId"));
                }
                Ok(Item::Invoke {
                    callee,
                    args: values,
                    signature,
                })
            }
            _ => {
                let atom = self.atom(node)?;
                if atom.kind == AtomKind::OptionalValue {
                    return Err(self.error(
                        node.offset,
                        "optional values require an optional named field",
                    ));
                }
                Ok(Item::Atom(atom))
            }
        }
    }

    fn named(&mut self, node: &Node) -> Result<Named, Error> {
        let (atom, mode) = match &node.kind {
            Kind::Call(name, args) if name == "optional" && args.len() == 1 => {
                let atom = self.atom(&args[0])?;
                if atom.kind != AtomKind::OptionalValue {
                    return Err(
                        self.error(node.offset, "optional named fields require optional(Value)")
                    );
                }
                (atom, Mode::Optional)
            }
            Kind::Call(name, args) if name == "default" && args.len() == 2 => {
                let atom = self.atom(&args[0])?;
                let Kind::Number(value) = args[1].kind else {
                    return Err(
                        self.error(args[1].offset, "text default must be an unsigned integer")
                    );
                };
                if !matches!(&atom.kind, AtomKind::Scalar(ty) if numeric_default(&PropertyType::Named(ty.clone()), value))
                {
                    return Err(self.error(
                        node.offset,
                        "text default is incompatible with the field type",
                    ));
                }
                (atom, Mode::Default(value))
            }
            _ => {
                let atom = self.atom(node)?;
                if atom.kind == AtomKind::OptionalValue || atom.kind == AtomKind::Values {
                    return Err(self.error(
                        node.offset,
                        "named fields require a bounded atom or explicit optional value",
                    ));
                }
                (atom, Mode::Required)
            }
        };
        let key = atom
            .path
            .rsplit('.')
            .next()
            .expect("field path has a component")
            .into();
        Ok(Named { atom, key, mode })
    }
}

fn list<'a>(node: &'a Node, source: &str) -> Result<&'a [Node], Error> {
    match &node.kind {
        Kind::List(items) => Ok(items),
        _ => Err(Error::at(source, node.offset, "expected text item list")),
    }
}

fn path<'a>(node: &'a Node, source: &str) -> Result<&'a str, Error> {
    match &node.kind {
        Kind::Name(path) => Ok(path),
        _ => Err(Error::at(
            source,
            node.offset,
            "expected logical field path",
        )),
    }
}

fn simple_scalar(ty: &str) -> bool {
    matches!(
        ty,
        "u8" | "u32"
            | "u64"
            | "i32"
            | "bool"
            | "FuncId"
            | "SigId"
            | "Intrinsic"
            | "IntCC"
            | "FloatCC"
            | "StackSlot"
    )
}

fn float_result(signature: &TypeDef) -> bool {
    let Some(result) = signature
        .results
        .patterns()
        .and_then(|results| results.first())
    else {
        return false;
    };
    match result {
        Pattern::Exact(ty) => matches!(ty.as_str(), "F32" | "F64"),
        Pattern::Class(class) | Pattern::Bind(_, class) => class == "ScalarFloat",
        Pattern::Same(slot) => {
            let operands = match &signature.operands {
                TypeList::Fixed(operands) | TypeList::Variadic(operands) => operands.as_slice(),
                TypeList::Signature => &[],
            };
            operands.iter().any(|pattern| {
                matches!(pattern, Pattern::Bind(other, class) if slot == other && class == "ScalarFloat")
            })
        }
        _ => false,
    }
}

fn single_token(kind: &AtomKind) -> bool {
    matches!(
        kind,
        AtomKind::Value | AtomKind::Integer | AtomKind::Float | AtomKind::Bytes
    ) || matches!(kind, AtomKind::Scalar(ty) if ty != "SigId")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Param;
    use crate::records::RecordField;

    fn op(params: &[(&str, &str)], text: Option<&str>) -> Op {
        let text = text.map(|text| {
            let source = format!("format Holder {{ text: {text} }}");
            crate::syntax::parse(&source)
                .unwrap()
                .pop()
                .unwrap()
                .fields
                .remove("text")
                .unwrap()
        });
        Op {
            offset: 0,
            name: "Test".into(),
            mnemonic: "test".into(),
            format: "Test".into(),
            signature: TypeDef {
                operands: TypeList::Fixed(vec![]),
                results: TypeList::Fixed(vec![]),
                relations: vec![],
            },
            params: params
                .iter()
                .map(|(name, kind)| Param {
                    name: (*name).into(),
                    kind: match *kind {
                        "value" => ParamKind::Value,
                        "values" => ParamKind::Values,
                        "successor" => ParamKind::Successor,
                        "successors" => ParamKind::Successors,
                        ty => ParamKind::Property(ty.into()),
                    },
                })
                .collect(),
            packing: BTreeMap::new(),
            signature_source: None,
            text,
            traits: vec![],
            memory: "NONE".into(),
            constraints: vec![],
            identity: None,
            absorbing: None,
            semantics: None,
        }
    }

    fn record() -> RecordDef {
        RecordDef {
            name: "Config".into(),
            storage: "VectorMemExtId".into(),
            fields: vec![
                RecordField {
                    name: "mask".into(),
                    ty: PropertyType::Named("Value".into()),
                    default: None,
                },
                RecordField {
                    name: "evl".into(),
                    ty: PropertyType::Optional("Value".into()),
                    default: Some(DefaultValue::None),
                },
                RecordField {
                    name: "scale".into(),
                    ty: PropertyType::Named("u8".into()),
                    default: Some(DefaultValue::Number(1)),
                },
                RecordField {
                    name: "flags".into(),
                    ty: PropertyType::Named("MemFlags".into()),
                    default: Some(DefaultValue::Empty),
                },
            ],
        }
    }

    #[test]
    fn implicit_text_uses_logical_order_and_requires_complete_projections() {
        let params = [("lhs", "value"), ("rhs", "value")];
        let schema = compile(&op(&params, None), &[], "").unwrap();
        assert!(matches!(&schema.args[0], Item::Atom(Atom { path, .. }) if path == "lhs"));
        assert!(matches!(&schema.args[1], Item::Atom(Atom { path, .. }) if path == "rhs"));
        for text in [
            "Text { args: [lhs] }",
            "Text { args: [lhs, lhs] }",
            "Text { args: [lhs, missing] }",
            "Other { args: [lhs, rhs] }",
            "Text { args: [lhs, rhs], extra: lhs }",
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[], "").is_err(),
                "{text}"
            );
        }
    }

    #[test]
    fn codecs_require_their_declared_logical_property_types() {
        for (codec, ty, expected) in [
            ("integer", "u64", AtomKind::Integer),
            ("float", "u64", AtomKind::Float),
            ("bytes", "Bytes", AtomKind::Bytes),
        ] {
            let text = format!("Text {{ args: [{codec}(arg)] }}");
            let mut operation = op(&[("arg", ty)], Some(&text));
            operation.signature.results =
                TypeList::Fixed(vec![Pattern::Class("ScalarFloat".into())]);
            let schema = compile(&operation, &[], "").unwrap();
            assert!(matches!(&schema.args[0], Item::Atom(atom) if atom.kind == expected));
            assert!(compile(&op(&[("arg", "value")], Some(&text)), &[], "").is_err());
        }
        assert!(compile(&op(&[("flags", "MemFlags")], None), &[], "").is_err());
        assert!(compile(&op(&[("bytes", "Bytes")], None), &[], "").is_err());
    }

    #[test]
    fn record_leaves_are_consumed_or_initialized_from_defaults() {
        let params = [("args", "values"), ("ext", "Config")];
        let text = "Text { args: [args], named: [ext.mask, optional(ext.evl)], flags: ext.flags }";
        let schema = compile(&op(&params, Some(text)), &[record()], "").unwrap();
        assert_eq!(schema.flags.as_deref(), Some("ext.flags"));
        assert_eq!(schema.named[0].key, "mask");
        assert!(matches!(schema.named[1].mode, Mode::Optional));
        assert!(
            matches!(&schema.defaults[..], [(path, DefaultValue::Number(1))] if path == "ext.scale")
        );
        for text in [
            "Text { args: [args] }",
            "Text { args: [args], named: [ext.unknown] }",
            "Text { args: [args], named: [ext.mask], flags: ext.scale }",
            "Text { args: [args], named: [ext.mask, optional(ext.scale)] }",
            "Text { args: [args], named: [ext.mask, default(ext.scale, 256)] }",
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[record()], "").is_err(),
                "{text}"
            );
        }
    }

    #[test]
    fn floating_atoms_require_a_statically_scalar_float_first_result() {
        let mut operation = op(&[("arg", "u64")], Some("Text { args: [float(arg)] }"));
        for result in [
            Pattern::Class("ScalarFloat".into()),
            Pattern::Exact("F32".into()),
            Pattern::Exact("F64".into()),
            Pattern::Bind(0, "ScalarFloat".into()),
        ] {
            operation.signature.results = TypeList::Fixed(vec![result]);
            compile(&operation, &[], "").unwrap();
        }
        for results in [
            TypeList::Fixed(vec![]),
            TypeList::Signature,
            TypeList::Fixed(vec![Pattern::Class("Float".into())]),
            TypeList::Fixed(vec![Pattern::Class("ScalarInteger".into())]),
            TypeList::Fixed(vec![Pattern::Exact("I32".into())]),
            TypeList::Fixed(vec![Pattern::Same(0)]),
        ] {
            operation.signature.results = results;
            assert!(compile(&operation, &[], "").is_err());
        }
        operation.signature.results = TypeList::Fixed(vec![Pattern::Same(0)]);
        operation.signature.operands =
            TypeList::Fixed(vec![Pattern::Bind(0, "ScalarFloat".into())]);
        compile(&operation, &[], "").unwrap();
        operation.signature.operands = TypeList::Fixed(vec![Pattern::Bind(0, "Float".into())]);
        assert!(compile(&operation, &[], "").is_err());

        let definitions = [
            include_str!("../../../mir/defs/formats.ops"),
            include_str!("../../../mir/defs/mir.ops"),
        ]
        .join("\n");
        let bad = definitions.replacen(
            "op Fconst(@value: u64) -> (result: ScalarFloat)",
            "op Fconst(@value: u64) -> (result: ScalarInteger)",
            1,
        );
        assert_ne!(bad, definitions);
        let error = crate::compile_mir(&bad)
            .err()
            .expect("unparseable float projection");
        assert!(
            error.message.contains("scalar float first result"),
            "{error}"
        );
    }

    #[test]
    fn named_keys_and_consumption_are_unambiguous() {
        let params = [("mask", "value"), ("ext", "Config")];
        for text in [
            "Text { named: [mask, ext.mask] }",
            "Text { args: [mask], named: [ext.mask, ext.mask] }",
            "Text { args: [mask], named: [ext.mask, ext.evl] }",
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[record()], "").is_err(),
                "{text}"
            );
        }
        let params = [("value", "u32")];
        for text in [
            "Text { named: [optional(value)] }",
            "Text { named: [default(value, true)] }",
        ] {
            assert!(compile(&op(&params, Some(text)), &[], "").is_err());
        }
        let schema = compile(
            &op(&params, Some("Text { named: [default(value, 7)] }")),
            &[],
            "",
        )
        .unwrap();
        assert!(matches!(schema.named[0].mode, Mode::Default(7)));
    }

    #[test]
    fn variadic_and_compound_items_have_decodable_boundaries() {
        let params = [("callee", "value"), ("args", "values"), ("sig", "SigId")];
        let schema = compile(
            &op(&params, Some("Text { args: [invoke(callee, args, sig)] }")),
            &[],
            "",
        )
        .unwrap();
        assert!(matches!(
            &schema.args[0],
            Item::Invoke {
                signature: Some(_),
                ..
            }
        ));
        for text in [
            "Text { args: [callee, args, sig] }",
            "Text { args: [space(callee, args), sig] }",
            "Text { args: [space(callee, sig), args] }",
            "Text { args: [invoke(sig, args, callee)] }",
            "Text { args: [invoke(callee, sig, args)] }",
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[], "").is_err(),
                "{text}"
            );
        }
        let params = [("cc", "IntCC"), ("lhs", "value"), ("rhs", "value")];
        let schema = compile(
            &op(&params, Some("Text { args: [space(cc, lhs), rhs] }")),
            &[],
            "",
        )
        .unwrap();
        assert!(matches!(schema.args[0], Item::Space(..)));
    }
}
