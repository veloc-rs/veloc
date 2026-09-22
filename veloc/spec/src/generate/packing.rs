//! Typed projections between logical parameters and physical instruction fields.

use crate::Error;
use crate::model::{Binding, Definitions, Op, Param, ParamKind, TypeDef, TypeList};
use crate::storage::{Alternative as LayoutAlternative, FieldType, Format};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

/// Construct physical storage from already typed logical locals.
pub(crate) fn constructor(
    op: &Op,
    format: &Format,
    opcode: &str,
    local: impl Fn(&str) -> String,
) -> String {
    let write = crate::storage::compact::construction(op, format, opcode, local);
    format!("move |writer: crate::InstWriter<'_>| {}", write.emit())
}

pub(crate) struct Alternative {
    pub op: Op,
    pub format: Format,
    pub targets: Vec<usize>,
}

/// Resolve and check each alternate once, shared by text and validation output.
pub(crate) fn prepare_alternatives(
    defs: &Definitions,
    formats: &[usize],
    source: &str,
) -> Result<Vec<Alternative>, Error> {
    let mut alternatives = Vec::new();
    let mut expressions = defs.expressions.clone();
    for alt in &defs.storage.alternatives {
        let targets = defs
            .storage
            .formats
            .iter()
            .enumerate()
            .filter_map(|(index, format)| alt.formats.contains(&format.name).then_some(index))
            .collect::<Vec<_>>();
        let base = defs
            .ops
            .iter()
            .zip(formats)
            .find(|(_, index)| targets.contains(index));
        let Some((base, _)) = base else {
            if alt.constraints.is_some() {
                return Err(Error::at(
                    source,
                    alt.text.offset,
                    "constraint layout has no operation",
                ));
            }
            continue;
        };
        let (mut op, format) = alternate(base, alt, source)?;
        op.inputs = crate::storage::compact::inputs(&op, &format);
        op.constraints = expressions.verify(
            source,
            alt.constraints.clone(),
            &op.params,
            Some(&op.signature),
            &BTreeMap::new(),
            crate::model::Vocabulary {
                types: &defs.types,
                data: &defs.data,
                encodings: &defs.encodings,
            },
        )?;
        alternatives.push(Alternative {
            op,
            format,
            targets,
        });
    }

    Ok(alternatives)
}

/// An alternate storage layout exposes its own text-facing fields. Pool handles
/// become structured properties, while its primary list retains canonical arity.
fn alternate(op: &Op, alt: &LayoutAlternative, source: &str) -> Result<(Op, Format), Error> {
    let mut params = Vec::new();
    let mut packing = BTreeMap::new();
    for field in &alt.fields {
        let FieldType::Named(ty) = &field.ty else {
            return Err(Error::at(
                source,
                alt.text.offset,
                "alternate text fields must use named storage types",
            ));
        };
        let (kind, binding) = match ty.as_str() {
            "Opcode" => continue,
            _ if field.policy.references.is_operand() => {
                (ParamKind::Value, Binding::Name(field.name.clone()))
            }
            _ if field.policy.references.is_operands() => {
                (ParamKind::Values, Binding::Name(field.name.clone()))
            }
            _ if field.policy.references.is_edge() => {
                (ParamKind::Successor, Binding::Name(field.name.clone()))
            }
            _ if field.policy.references.is_edges() => {
                (ParamKind::Successors, Binding::Name(field.name.clone()))
            }
            "ConstantPoolId" => (
                ParamKind::Property("Bytes".into()),
                Binding::Pool(field.name.clone()),
            ),
            _ => (
                ParamKind::Property(ty.clone()),
                Binding::Name(field.name.clone()),
            ),
        };
        params.push(Param {
            moves: false,
            name: field.name.clone(),
            kind,
        });
        packing.insert(field.name.clone(), binding);
    }
    Ok((
        Op {
            declaration: op.declaration.clone(),
            offset: alt.text.offset,
            name: op.name.clone(),
            mnemonic: op.mnemonic.clone(),
            meta: op.meta.clone(),
            format: alt.name.clone(),
            signature: TypeDef {
                operands: TypeList::Fixed(Vec::new()),
                results: TypeList::Fixed(Vec::new()),
            },
            params,
            inputs: Default::default(),
            projection: crate::model::Projection::Packed(packing),
            signature_source: None,
            text: Some(alt.text.clone()),
            traits: BTreeSet::new(),
            queries: BTreeMap::new(),
            constraints: Vec::new(),
            identity: None,
            absorbing: None,
            semantics: None,
        },
        Format {
            name: alt.name.clone(),
            arity: None,
            fields: alt.fields.clone(),
        },
    ))
}

pub(crate) struct Builder {
    inferred: Option<Vec<crate::types::rules::ResultExpr>>,
}

pub(crate) fn prepare_builder(op: &Op, source: &str) -> Result<Option<Builder>, Error> {
    let fail = |message| Error::at(source, op.offset, message);
    let ty = &op.signature;
    let results = ty.results.patterns();
    if results.is_none() && op.signature_source.is_none() {
        return Ok(None);
    }
    if op
        .params
        .iter()
        .any(|param| matches!(param.kind, ParamKind::Successor | ParamKind::Successors))
    {
        return Ok(None);
    }
    let name = op.method_name();
    crate::model::identifier(source, op.offset, &name)?;
    if matches!(
        name.as_str(),
        "block"
            | "builder"
            | "param"
            | "params"
            | "value_type"
            | "emit"
            | "insert"
            | "constant"
            | "dense_const"
    ) {
        return Err(fail(format!(
            "operation `{}` conflicts with an InstCursor method",
            op.mnemonic
        )));
    }
    let inferred = crate::types::rules::result_exprs(ty);
    let typed = inferred.is_none() && op.signature_source.is_none();
    if typed && results.unwrap().len() != 1 {
        return Err(fail(
            "field builder requires exactly one explicit result".into(),
        ));
    }
    let mut names = BTreeSet::new();
    if typed {
        names.insert("ty".to_owned());
    }
    for param in &op.params {
        if !names.insert(param.name.clone()) {
            return Err(fail(format!(
                "conflicting generated parameter `{}`",
                param.name
            )));
        }
    }
    Ok(Some(Builder { inferred }))
}

pub(crate) fn builder(
    op: &Op,
    format: &Format,
    builder: &Builder,
    rust: &crate::model::records::RustTypes,
) -> String {
    let name = op.method_name();
    let inferred = &builder.inferred;
    let typed = inferred.is_none() && op.signature_source.is_none();
    let mut params = String::from("&mut self");
    for param in &op.params {
        let ty = match &param.kind {
            ParamKind::Value => "crate::Value".to_owned(),
            ParamKind::Values => "&[crate::Value]".to_owned(),
            ParamKind::Property(ty) if ty == "Bytes" => "alloc::vec::Vec<u8>".into(),
            ParamKind::Property(ty) => rust.qualified(ty),
            ParamKind::Successor | ParamKind::Successors => {
                unreachable!("contextual builder was excluded")
            }
        };
        write!(params, ", {}: {ty}", param.name).unwrap();
    }
    if typed {
        params.push_str(", ty: crate::Type");
    }
    let constructor = crate::generate::packing::constructor(op, format, &op.name, str::to_owned);
    if let Some(source) = &op.signature_source {
        use crate::model::SignatureSource;
        let signature = match source {
            SignatureSource::Function(param) => format!("self.decls[{param}].signature"),
            SignatureSource::Signature(param) => param.clone(),
            SignatureSource::Value(param) => format!(
                "self.value_type({param}).as_callable().expect(\"call requires a callable value\").0"
            ),
        };
        // Borrow the external signature table, not the cursor, so insertion can
        // mutably borrow the editor without copying the return type slice.
        return format!(
            "    /// Build `{}` from its declared signature without validating arguments.\n    pub fn {name}({params}) -> crate::Inst {{\n        let signatures = self.signatures;\n        let returns = signatures[{signature}].returns();\n        self.insert({constructor}, returns)\n    }}\n",
            op.mnemonic
        );
    }
    let results = op
        .signature
        .results
        .patterns()
        .expect("prepared builder results");
    let result_types = if let Some(inferred) = inferred {
        let operands = op
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Value)
            .collect::<Vec<_>>();
        inferred.iter().map(|r| match r {
            crate::types::rules::ResultExpr::Property(name) => format!("{name}.ty()"),
            crate::types::rules::ResultExpr::Exact(ty) => crate::types::rust_type(ty),
            crate::types::rules::ResultExpr::Operand(index) => format!("self.value_type({})", operands[*index].name),
            crate::types::rules::ResultExpr::Element(index) => format!("self.value_type({}).as_vector().expect(\"result element type requires a vector operand\").element_type().as_type()", operands[*index].name),
        }).collect::<Vec<_>>().join(", ")
    } else {
        "ty".into()
    };
    let (ret, body) = match results.len() {
        0 => (String::new(), "self.insert(data, &types);".to_owned()),
        1 => (
            " -> crate::Value".to_owned(),
            "let [result] = self.emit(data, types);\n        result".to_owned(),
        ),
        count => {
            let types = vec!["crate::Value"; count].join(", ");
            let names = (0..count)
                .map(|i| format!("result{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            (
                format!(" -> ({types})"),
                format!("let [{names}] = self.emit(data, types);\n        ({names})"),
            )
        }
    };
    format!(
        "    /// Build `{}` without validating its type contract.\n    pub fn {name}({params}){ret} {{\n        let (data, types) = ({constructor}, [{result_types}]);\n        {body}\n    }}\n",
        op.mnemonic
    )
}
