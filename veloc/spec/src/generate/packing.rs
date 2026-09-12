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
    let mut fields = Vec::new();
    let mut setup = String::new();
    for field in &format.fields {
        let value = if matches!(&field.ty, FieldType::Named(ty) if ty == "Opcode") {
            format!("crate::Opcode::{opcode}")
        } else {
            match &op.bindings()[&field.name] {
                Binding::Name(name) => {
                    let value = local(name);
                    if field.policy.references.is_edge() {
                        format!("({value}).as_view()")
                    } else {
                        value
                    }
                }
                Binding::Array(args) => {
                    let args = args
                        .iter()
                        .map(|arg| {
                            let Binding::Name(name) = arg else {
                                unreachable!("checked array binding")
                            };
                            local(name)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{args}]")
                }
                Binding::Pool(name) => {
                    let value = local(name);
                    let ty = field.rust.clone();
                    let var = format!("_pooled_{}", field.name);
                    writeln!(setup, "let {var} = {ty}::insert(writer.dfg, {value});").unwrap();
                    var
                }
                Binding::Table { cases, default } => {
                    format!(
                        "({}).iter().map(crate::BlockCall::as_view).chain(core::iter::once(({}).as_view()))",
                        local(cases),
                        local(default)
                    )
                }
            }
        };
        fields.push(value);
    }
    format!(
        "move |writer: crate::InstWriter<'_>| {{ {setup} writer.{}({}) }}",
        crate::storage::constructor_name(&format.name),
        fields.join(", ")
    )
}

/// Recover logical locals from physical values. Records, byte buffers and
/// variadic lists are borrowed; the caller selects its error representation.
pub(crate) fn projections(
    op: &Op,
    format: &Format,
    dfg: &str,
    field: impl Fn(&str) -> String,
    required: impl Fn(String) -> String,
) -> Vec<(String, String)> {
    let mut locals = Vec::new();
    for storage in &format.fields {
        if matches!(&storage.ty, FieldType::Named(ty) if ty == "Opcode") {
            continue;
        }
        let value = field(&storage.name);
        match &op.bindings()[&storage.name] {
            Binding::Name(name) => {
                locals.push((name.clone(), value));
            }
            Binding::Array(args) => {
                let value = format!("({value})");
                for (index, arg) in args.iter().enumerate() {
                    let Binding::Name(name) = arg else {
                        unreachable!("checked array binding")
                    };
                    locals.push((name.clone(), format!("{value}[{index}]")));
                }
            }
            Binding::Pool(name) => {
                let ty = storage.rust.clone();
                let value = required(format!("{ty}::get({value}, {dfg})"));
                locals.push((name.clone(), value));
            }
            Binding::Table { cases, default } => {
                let split = required(format!("({value}).split_last()"));
                locals.push((cases.clone(), format!("({split}).1")));
                locals.push((default.clone(), format!("({split}).0")));
            }
        }
    }
    locals
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
            if !alt.constraints.is_empty() {
                return Err(Error::at(
                    source,
                    alt.text.offset,
                    "constraint layout has no operation",
                ));
            }
            continue;
        };
        let (mut op, format) = alternate(base, alt, source)?;
        op.constraints = expressions.verify(
            source,
            alt.constraints.clone(),
            &op.params,
            Some(&op.signature),
            &BTreeMap::new(),
            crate::model::Vocabulary {
                types: &defs.types,
                data: &defs.data,
                builtins: &defs.builtins,
                comparisons: &defs.comparisons,
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
            projection: crate::model::Projection::Packed(packing),
            signature_source: None,
            text: Some(alt.text.clone()),
            traits: Vec::new(),
            memory: crate::model::builtins::Effect::Known(Vec::new()),
            interfaces: BTreeMap::new(),
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

pub(crate) fn accessors(defs: &Definitions) -> String {
    let mut output = String::from(
        "impl<'a> crate::InstView<'a> {\n    /// Visit outgoing block calls in storage order, preserving edge arguments and duplicates.\n    pub fn visit_successors(&self, mut f: impl FnMut(crate::Successor<'a>)) {\nself.try_visit_successors::<core::convert::Infallible>(|edge| { f(edge); Ok(()) }).unwrap_or_else(|never| match never {});\n}\n/// Visit successors in storage order, stopping at the first error.\npub fn try_visit_successors<E>(&self, mut f: impl FnMut(crate::Successor<'a>) -> core::result::Result<(), E>) -> core::result::Result<(), E> {\n        match self {\n",
    );
    for format in &defs.storage.formats {
        let edges: Vec<_> = format.fields.iter().filter(|field| {
            matches!(&field.ty, FieldType::Named(ty) if matches!(ty.as_str(), "BlockCall" | "JumpTable"))
        }).collect();
        if edges.is_empty() {
            continue;
        }
        let bindings = edges
            .iter()
            .enumerate()
            .map(|(index, field)| format!("{}: edge{index}", field.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "            crate::InstView::{} {{ {bindings}, .. }} => {{",
            format.name
        )
        .unwrap();
        for (index, field) in edges.iter().enumerate() {
            if matches!(&field.ty, FieldType::Named(ty) if ty == "JumpTable") {
                writeln!(
                    output,
                    "                for call in edge{index}.iter() {{ f(call)?; }}"
                )
                .unwrap();
            } else {
                writeln!(output, "                f(*edge{index})?;").unwrap();
            }
        }
        output.push_str("            },\n");
    }
    output.push_str("            _ => {},\n        }\nOk(())\n    }\n}\n");
    output
}

pub(crate) struct Builder {
    inferred: Option<Vec<crate::types::rules::ResultExpr>>,
}

pub(crate) fn prepare_builder(op: &Op, source: &str) -> Result<Option<Builder>, Error> {
    let fail = |message| Error::at(source, op.offset, message);
    let ty = &op.signature;
    let Some(results) = ty.results.patterns() else {
        // Signature- and context-selected results need the module's help.
        return Ok(None);
    };
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
            | "insert_inferred"
            | "insert"
            | "constant"
            | "dense_const"
    ) {
        return Err(fail(format!(
            "operation `{}` conflicts with an InstBuilder method",
            op.mnemonic
        )));
    }
    let inferred = crate::types::rules::result_exprs(ty);
    let typed = inferred.is_none();
    if typed && results.len() != 1 {
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
    let results = op
        .signature
        .results
        .patterns()
        .expect("prepared builder results");
    let inferred = &builder.inferred;
    let typed = inferred.is_none();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_immutable_bytes_are_interned() {
        let defs = crate::Source::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mir/defs/module.ops"),
        )
        .unwrap()
        .parse()
        .unwrap();
        for (name, logical, pooled) in [
            ("PtrIndex", "imm", false),
            ("LoadStride", "mem", false),
            ("Vconst", "value", false),
            ("Shuffle", "mask", true),
        ] {
            let op = defs.ops.iter().find(|op| op.name == name).unwrap();
            let format = defs
                .storage
                .formats
                .iter()
                .find(|f| f.name == op.format)
                .unwrap();
            let packed = constructor(op, format, &op.name, str::to_owned);
            assert_eq!(packed.contains("::insert("), pooled, "{packed}");
            let locals = projections(op, format, "dfg", str::to_owned, |value| {
                format!("{value}.ok_or(invalid)?")
            });
            let (_, expr) = locals.iter().find(|(name, _)| name == logical).unwrap();
            assert_eq!(expr.contains("::get("), pooled, "{expr}");
        }
    }

    #[test]
    fn jump_table_projection_splits_default_from_cases() {
        let defs = crate::Source::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mir/defs/module.ops"),
        )
        .unwrap()
        .parse()
        .unwrap();
        let op = defs.ops.iter().find(|op| op.name == "BrTable").unwrap();
        let format = defs
            .storage
            .formats
            .iter()
            .find(|format| format.name == op.format)
            .unwrap();
        assert!(
            constructor(op, format, &op.name, str::to_owned)
                .contains("chain(core::iter::once((default).as_view()))")
        );
        let locals = projections(op, format, "dfg", str::to_owned, |value| {
            format!("{value}.ok_or(invalid)?")
        });
        assert!(
            locals
                .iter()
                .any(|(name, expr)| name == "cases" && expr.ends_with(").1"))
        );
        assert!(locals.iter().any(|(name, expr)| name == "default"
            && expr.starts_with("(")
            && expr.ends_with(").0")));
    }
}
