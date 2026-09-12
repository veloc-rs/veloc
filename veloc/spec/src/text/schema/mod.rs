//! Checked text projections over logical parameters, not storage layouts.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::model::data::Value;
use crate::model::records::{PropertyType, RecordDef};
use crate::model::{Op, ParamKind, Pattern, TypeDef, TypeList};
use crate::syntax::Kind;

mod template;

#[derive(Debug)]
pub(super) struct Schema {
    pub args: Vec<Item>,
    pub named: Vec<Named>,
    pub flags: Option<String>,
    pub bindings: Vec<(String, Value)>,
}

#[derive(Debug)]
pub(super) enum Item {
    Atom(Atom),
    Space(Box<Item>, Box<Item>),
    Invoke {
        callee: Atom,
        args: Atom,
        signature: CallSignature,
    },
}

#[derive(Debug)]
pub(super) enum CallSignature {
    Value,
    Field(Atom),
    Function,
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
}

struct Checker<'a> {
    source: &'a str,
    leaves: BTreeMap<String, AtomKind>,
    used: BTreeSet<String>,
    typed: BTreeSet<String>,
}

pub(super) fn compile(
    op: &Op,
    records: &[RecordDef],
    source: &str,
    types: &crate::types::Types,
) -> Result<Schema, Error> {
    let mut checker = Checker {
        source,
        leaves: BTreeMap::new(),
        used: BTreeSet::new(),
        typed: BTreeSet::new(),
    };
    for param in &op.params {
        if let ParamKind::Property(ty) = &param.kind
            && let Some(record) = records.iter().find(|record| record.name == *ty)
        {
            for field in &record.fields {
                let kind = match &field.ty {
                    PropertyType::Named(_) if field.policy.references.is_operand() => {
                        AtomKind::Value
                    }
                    PropertyType::Named(ty) => AtomKind::Scalar(ty.clone()),
                    PropertyType::Optional(_) if field.policy.references.is_operand() => {
                        AtomKind::OptionalValue
                    }
                    _ => return Err(checker.error(op.offset, "unsupported optional property")),
                };
                checker
                    .leaves
                    .insert(format!("{}.{}", param.name, field.name), kind);
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
        checker.leaves.insert(param.name.clone(), kind);
    }

    let mut schema = Schema {
        args: Vec::new(),
        named: Vec::new(),
        flags: None,
        bindings: Vec::new(),
    };
    if let Some(node) = &op.text {
        let Kind::Text(text) = &node.kind else {
            return Err(checker.error(node.offset, "expected a quoted text template"));
        };
        template::compile(&mut checker, &mut schema, text, node.offset)?;
    } else {
        for param in &op.params {
            schema
                .args
                .push(Item::Atom(checker.atom(&param.name, None, op.offset)?));
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
    for ty in &checker.typed {
        let description = match ty.as_str() {
            "Float" => "scalar float",
            "Int" => "scalar integer",
            "VectorConst" => "vector",
            _ => unreachable!("typed text property"),
        };
        let allowed = types.property_types(ty).expect("typed text property");
        if !result_in_set(&op.signature, types, &allowed) {
            return Err(checker.error(
                op.offset,
                format!(
                    "{} text atoms require a {description} first result",
                    description.strip_prefix("scalar ").unwrap_or(description)
                ),
            ));
        }
    }
    for path in checker.leaves.keys() {
        if !checker.used.contains(path) {
            return Err(Error::at(
                source,
                op.offset,
                format!("text projection does not consume `{path}`"),
            ));
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
        Ok(leaf.clone())
    }

    fn atom(&mut self, path: &str, codec: Option<&str>, offset: usize) -> Result<Atom, Error> {
        let kind = self.consume(path, offset)?;
        let kind = match (codec, kind) {
            (Some("integer"), AtomKind::Scalar(ty)) if ty == "u64" => AtomKind::Integer,
            (None, AtomKind::Scalar(ty))
                if matches!(ty.as_str(), "Float" | "Int" | "VectorConst") =>
            {
                self.typed.insert(ty.clone());
                AtomKind::Scalar(ty)
            }
            (Some("bytes"), AtomKind::Scalar(ty)) if ty == "Bytes" => AtomKind::Bytes,
            (Some(codec), _) => {
                return Err(self.error(
                    offset,
                    format!("text codec `{codec}` is incompatible with `{path}`"),
                ));
            }
            (None, AtomKind::Scalar(ty)) if !simple_scalar(&ty) => {
                return Err(self.error(
                    offset,
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
            | "Int"
    )
}

fn result_in_set(
    signature: &TypeDef,
    types: &crate::types::Types,
    allowed: &crate::types::TypeSet,
) -> bool {
    let Some(result) = signature
        .results
        .patterns()
        .and_then(|results| results.first())
    else {
        return false;
    };
    let accepts = |set: &crate::types::TypeSet| set.subset_of(allowed);
    match result {
        Pattern::Exact(ty) => accepts(&types.exact[ty]),
        Pattern::Set(set) | Pattern::Bind(_, set) | Pattern::Property(_, set) => accepts(set),
        Pattern::Same(slot) => {
            let operands = match &signature.operands {
                TypeList::Fixed(operands) | TypeList::Variadic(operands) => operands.as_slice(),
                TypeList::Signature => &[],
            };
            operands.iter().any(|pattern| {
                matches!(pattern, Pattern::Bind(other, set) if slot == other && accepts(set))
            })
        }
        _ => false,
    }
}

fn single_token(kind: &AtomKind) -> bool {
    matches!(kind, AtomKind::Value | AtomKind::Integer | AtomKind::Bytes)
        || matches!(kind, AtomKind::Scalar(ty) if !matches!(ty.as_str(), "SigId" | "FuncId" | "VectorConst"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Param;
    use crate::model::records::RecordField;

    fn compile(op: &Op, records: &[RecordDef], source: &str) -> Result<Schema, Error> {
        super::compile(op, records, source, &crate::fixtures::types())
    }

    fn op(params: &[(&str, &str)], text: Option<&str>) -> Op {
        let text = text.map(|text| {
            let source = format!("fixture Holder {{ text: {text} }}");
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
            meta: crate::model::data::Value::Record("OpInfo".into(), Default::default()),
            format: "Test".into(),
            signature: TypeDef {
                operands: TypeList::Fixed(vec![]),
                results: TypeList::Fixed(vec![]),
            },
            params: params
                .iter()
                .map(|(name, kind)| Param {
                    moves: false,
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
            projection: crate::model::Projection::Packed(BTreeMap::new()),
            signature_source: None,
            text,
            traits: vec![],
            memory: crate::model::builtins::Effect::Known(Vec::new()),
            interfaces: BTreeMap::new(),
            constraints: vec![],
            identity: None,
            absorbing: None,
            semantics: None,
        }
    }

    fn record() -> RecordDef {
        RecordDef {
            name: "Config".into(),
            fields: vec![
                RecordField {
                    policy: crate::fixtures::types_policy("Value"),
                    name: "mask".into(),
                    ty: PropertyType::Named("Value".into()),
                    rust: "crate::Value".into(),
                },
                RecordField {
                    policy: crate::fixtures::types_policy("Value"),
                    name: "evl".into(),
                    ty: PropertyType::Optional("Value".into()),
                    rust: "Option<crate::Value>".into(),
                },
                RecordField {
                    policy: crate::model::records::Policy::default(),
                    name: "scale".into(),
                    ty: PropertyType::Named("u8".into()),
                    rust: "u8".into(),
                },
                RecordField {
                    policy: crate::model::records::Policy::default(),
                    name: "flags".into(),
                    ty: PropertyType::Named("MemFlags".into()),
                    rust: "MemFlags".into(),
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
            r#""{lhs}""#,
            r#""{lhs}, {lhs}""#,
            r#""{lhs}, {missing}""#,
            r#""{lhs}{rhs}""#,
            r#""{lhs}; {rhs}""#,
            r#""{lhs}, {rhs""#,
            r#""{lhs}, {rhs},""#,
            r#""{lhs}, rhs={rhs} trailing""#,
            r#""left={lhs}, {rhs}""#,
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
            ("bytes", "Bytes", AtomKind::Bytes),
        ] {
            let text = format!(r#""{{arg:{codec}}}""#);
            let mut operation = op(&[("arg", ty)], Some(&text));
            operation.signature.results =
                TypeList::Fixed(vec![Pattern::Set(crate::fixtures::set("ScalarFloat"))]);
            let schema = compile(&operation, &[], "").unwrap();
            assert!(matches!(&schema.args[0], Item::Atom(atom) if atom.kind == expected));
            assert!(compile(&op(&[("arg", "value")], Some(&text)), &[], "").is_err());
        }
        assert!(compile(&op(&[("flags", "MemFlags")], None), &[], "").is_err());
        assert!(compile(&op(&[("bytes", "Bytes")], None), &[], "").is_err());
    }

    #[test]
    fn record_leaves_require_explicit_text_or_fixed_bindings() {
        let params = [("args", "values"), ("ext", "Config")];
        let text = r#""{ext.scale=1} {.ext.flags} {args}, mask={ext.mask}[, evl={ext.evl}]""#;
        let schema = compile(&op(&params, Some(text)), &[record()], "").unwrap();
        assert_eq!(schema.flags.as_deref(), Some("ext.flags"));
        assert_eq!(schema.named[0].key, "mask");
        assert!(matches!(schema.named[1].mode, Mode::Optional));
        assert!(matches!(&schema.bindings[..], [(path, Value::Number(1))] if path == "ext.scale"));
        for text in [
            r#""{args}""#,
            r#""{args}, unknown={ext.unknown}""#,
            r#""{.ext.scale} {args}, mask={ext.mask}""#,
            r#""{args}, mask={ext.mask}[, scale={ext.scale}]""#,
            r#""{args}, mask={ext.mask}, scale={ext.scale=256}""#,
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[record()], "").is_err(),
                "{text}"
            );
        }
    }

    #[test]
    fn floating_atoms_require_a_statically_scalar_float_first_result() {
        let mut operation = op(&[("arg", "Float")], None);
        for result in [
            Pattern::Set(crate::fixtures::set("ScalarFloat")),
            Pattern::Exact("F32".into()),
            Pattern::Exact("F64".into()),
            Pattern::Bind(0, crate::fixtures::set("ScalarFloat")),
        ] {
            operation.signature.results = TypeList::Fixed(vec![result]);
            compile(&operation, &[], "").unwrap();
        }
        for results in [
            TypeList::Fixed(vec![]),
            TypeList::Signature,
            TypeList::Fixed(vec![Pattern::Set(crate::fixtures::set("Float"))]),
            TypeList::Fixed(vec![Pattern::Set(crate::fixtures::set("ScalarInteger"))]),
            TypeList::Fixed(vec![Pattern::Exact("I32".into())]),
            TypeList::Fixed(vec![Pattern::Same(0)]),
        ] {
            operation.signature.results = results;
            assert!(compile(&operation, &[], "").is_err());
        }
        operation.signature.results = TypeList::Fixed(vec![Pattern::Same(0)]);
        operation.signature.operands =
            TypeList::Fixed(vec![Pattern::Bind(0, crate::fixtures::set("ScalarFloat"))]);
        compile(&operation, &[], "").unwrap();
        operation.signature.operands =
            TypeList::Fixed(vec![Pattern::Bind(0, crate::fixtures::set("Float"))]);
        assert!(compile(&operation, &[], "").is_err());

        let definitions = [
            include_str!("../../../../mir/defs/formats.ops"),
            include_str!("../../../../mir/defs/mir.ops"),
        ]
        .join("\n")
        .lines()
        .filter(|line| !line.trim_start().starts_with("import "))
        .collect::<Vec<_>>()
        .join("\n");
        let bad = definitions.replacen(
            "op Fconst(value: Float) -> type(value)",
            "op Fconst(value: Float) -> ScalarInteger",
            1,
        );
        assert_ne!(bad, definitions);
        let error = crate::fixtures::compile(&bad)
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
            r#""mask={mask}, mask={ext.mask}""#,
            r#""{mask}, mask={ext.mask}, mask={ext.mask}""#,
            r#""{mask}, mask={ext.mask}, evl={ext.evl}""#,
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[record()], "").is_err(),
                "{text}"
            );
        }
        let params = [("value", "u32")];
        for text in [r#""[, value={value}]""#, r#""value={value=true}""#] {
            assert!(compile(&op(&params, Some(text)), &[], "").is_err());
        }
        let schema = compile(&op(&params, Some(r#""value={value}""#)), &[], "").unwrap();
        assert!(matches!(schema.named[0].mode, Mode::Required));
    }

    #[test]
    fn function_signature_projection_reuses_the_callee() {
        let params = [("target", "FuncId"), ("args", "values")];
        let schema = compile(
            &op(&params, Some(r#""{target}({args}) : {function(target)}""#)),
            &[],
            "",
        )
        .unwrap();
        assert!(matches!(
            &schema.args[0],
            Item::Invoke {
                signature: CallSignature::Function,
                ..
            }
        ));
        for text in [
            r#""{target}({args})""#,
            r#""{target}({args}) : {function(args)}""#,
            r#""{target}({args}) : {function()}""#,
            r#""{target}({args}) : {function(target, target)}""#,
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[], "").is_err(),
                "{text}"
            );
        }
        let params = [("target", "value"), ("args", "values")];
        assert!(
            compile(
                &op(&params, Some(r#""{target}({args}) : {function(target)}""#)),
                &[],
                "",
            )
            .is_err()
        );
    }

    #[test]
    fn variadic_and_compound_items_have_decodable_boundaries() {
        let params = [("callee", "value"), ("args", "values"), ("sig", "SigId")];
        let schema = compile(&op(&params, Some(r#""{callee}({args}) : {sig}""#)), &[], "").unwrap();
        assert!(matches!(
            &schema.args[0],
            Item::Invoke {
                signature: CallSignature::Field(_),
                ..
            }
        ));
        for text in [
            r#""{callee}, {args}, {sig}""#,
            r#""{callee} {args}, {sig}""#,
            r#""{callee} {sig}, {args}""#,
            r#""{sig}({args}) : {callee}""#,
            r#""{callee}({sig}) : {args}""#,
        ] {
            assert!(
                compile(&op(&params, Some(text)), &[], "").is_err(),
                "{text}"
            );
        }
        let params = [("cc", "IntCC"), ("lhs", "value"), ("rhs", "value")];
        let schema = compile(&op(&params, Some(r#""{cc} {lhs}, {rhs}""#)), &[], "").unwrap();
        assert!(matches!(schema.args[0], Item::Space(..)));
    }
}
