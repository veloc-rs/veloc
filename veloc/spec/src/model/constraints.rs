//! Verification statements and validator adapters over shared pure expressions.
use super::{
    Definitions, Op, ParamKind, Pattern,
    expr::{Emitter, Expr},
};
use std::collections::BTreeMap;
use std::fmt::Write;

pub(crate) struct Constraint {
    pub binding: Option<usize>,
    pub condition: Expr,
    pub text: String,
    pub type_only: bool,
}

impl Constraint {
    /// Bindings and checks preserve source order in every validator adapter.
    pub(crate) fn emit(&self, emitter: &Emitter<'_>, failure: &str) -> String {
        let value = emitter.term(&self.condition);
        if let Some(id) = self.binding {
            format!("let _v{id} = {value};\n")
        } else {
            format!("if !({value}) {{ {failure}; }}\n")
        }
    }
    pub(crate) fn redundant(&self) -> bool {
        self.binding.is_none() && self.condition.is_bool(true)
    }
}

/// Emit a contract against logical parameter bindings supplied by storage.
/// Contexts, local bindings and error propagation are identical for both IRs.
pub(crate) fn emit_checks(
    constraints: &[Constraint],
    emitter: &mut Emitter<'_>,
    contexts: &BTreeMap<&str, usize>,
    types_checked: bool,
    error: impl Fn(&str) -> String,
) -> String {
    let mut out = String::new();
    for constraint in constraints {
        if constraint.redundant()
            || (types_checked && constraint.type_only && constraint.binding.is_none())
        {
            continue;
        }
        if let Some(ty) = constraint.condition.context_type() {
            writeln!(out, "let _context = _ctx{};", contexts[ty]).unwrap();
        }
        let failure = error(&constraint.text);
        emitter.error = Some(failure.clone());
        out.push_str(&constraint.emit(emitter, &format!("return Err({failure})")));
    }
    out
}

/// Signature contracts are independent of the container owning values.
/// Hosts resolve a signature and compare their own value representation.
pub(crate) fn emit_signature(
    op: &Op,
    projections: &BTreeMap<String, String>,
    resolve: impl Fn(&super::SignatureSource, &str) -> String,
    validate: impl Fn(&str, &str, &str) -> String,
    results: &str,
) -> String {
    let Some(source) = &op.signature_source else {
        return String::new();
    };
    let (super::SignatureSource::Function(name)
    | super::SignatureSource::Signature(name)
    | super::SignatureSource::Value(name)) = source;
    let signature = resolve(source, &projections[name]);
    let args = &projections[&op
        .params
        .iter()
        .find(|p| p.kind == ParamKind::Values)
        .expect("checked signature arguments")
        .name];
    let args = format!("&({args})");
    format!(
        "let signature = {signature};\n{}\n{}\n",
        validate("value", &args, "signature.0"),
        validate("result", results, "signature.1")
    )
}

pub(crate) fn contexts<'a>(
    defs: &'a Definitions,
    alternatives: &'a [crate::generate::packing::Alternative],
) -> BTreeMap<&'a str, usize> {
    defs.ops
        .iter()
        .chain(alternatives.iter().map(|a| &a.op))
        .flat_map(|op| &op.constraints)
        .filter_map(|c| c.condition.context_type())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, ty)| (ty, i))
        .collect()
}

use crate::syntax::{Kind, Node};
pub(crate) fn describe(node: &Node) -> String {
    match &node.kind {
        Kind::Name(name) => name.clone(),
        Kind::Integer(n) => n.to_string(),
        Kind::Try(value) => format!("{}?", describe(value)),
        Kind::Number(n) => n.to_string(),
        Kind::Text(text) => format!("{text:?}"),
        Kind::Unary(op, value) => format!("{op}({})", describe(value)),
        Kind::Binary(op, lhs, rhs) => format!("({} {op} {})", describe(lhs), describe(rhs)),
        Kind::Member(receiver, name) => format!("{}.{name}", describe(receiver)),
        Kind::Method(receiver, name, args) => format!(
            "{}.{name}({})",
            describe(receiver),
            args.iter().map(describe).collect::<Vec<_>>().join(", ")
        ),
        Kind::Call(name, args) => format!(
            "{name}({})",
            args.iter().map(describe).collect::<Vec<_>>().join(", ")
        ),
        Kind::Query(name, _) => format!("query {name}"),
        Kind::Lambda(names, body) => format!("|{}| {}", names.join(", "), describe(body)),
        _ => "invalid constraint".into(),
    }
}

pub(crate) fn generate(
    defs: &Definitions,
    formats: &[usize],
    alternatives: &[crate::generate::packing::Alternative],
) -> String {
    // The caller supplies each concrete context once. Property validators and
    // alternate layouts share those same references; no adapter is constructed.
    let contexts = contexts(defs, alternatives);
    let mut groups = BTreeMap::<String, Vec<&str>>::new();
    for (op, &format) in defs.ops.iter().zip(formats) {
        let format = &defs.storage.formats[format];
        let body = emit_body(defs, op, format, true, &contexts);
        groups.entry(body).or_default().push(&op.name);
    }
    let params = contexts
        .iter()
        .map(|(ty, i)| format!(", _ctx{i}: &{ty}"))
        .collect::<String>();
    let mut out = format!(
        "// @generated by veloc-spec. Edit the .spec definitions instead.\nimpl Function {{\n    fn validate_constraints(&self, _dfg: &crate::dfg::DataFlowGraph, _module: &crate::ModuleData, _inst: Inst, data: &InstView<'_>, _operands: &[Type], _results: &[Type]{params}) -> Result<()> {{\n",
    );
    // Layout-specific auxiliary operands have contracts independent of opcode.
    for alternative in alternatives {
        let op = &alternative.op;
        let format = &alternative.format;
        if op.constraints.is_empty() {
            continue;
        }
        let body = emit_body(defs, op, format, false, &contexts);
        writeln!(
            out,
            "if matches!(data, InstView::{} {{ .. }}) {{ {body} }}",
            format.name
        )
        .unwrap();
    }
    out.push_str("        match data.opcode() {\n");
    for (body, names) in groups {
        let names = names
            .iter()
            .map(|name| format!("Opcode::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            out,
            "            {names} => {{\n{body}                Ok(())\n            }},"
        )
        .unwrap();
    }
    out.push_str("        }\n    }\n}\n");
    out
}

fn emit_body(
    defs: &Definitions,
    op: &Op,
    format: &crate::storage::Format,
    type_checks: bool,
    contexts: &BTreeMap<&str, usize>,
) -> String {
    let mut body = String::new();
    let mut storage_used = false;
    let error = |text: &str| format!("self.constraint_error(_inst, {text:?})");
    let projections = crate::model::access::projections(
        op,
        "_dfg",
        |name| {
            format!(
                "*_f{}",
                format.fields.iter().position(|f| f.name == name).unwrap()
            )
        },
        |v| format!("{v}.ok_or_else(|| {})?", error("missing property storage")),
    )
    .into_iter()
    .collect();
    let mut emitter = Emitter::types(op, projections, "_operands", "_results");
    emitter.prefix = "crate::inst::";
    emitter.dfg = "_dfg";
    for (index, pattern) in op
        .signature
        .results
        .patterns()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        if let Pattern::Property(name, _) = pattern {
            let value = &emitter.projections[name];
            writeln!(body, "if _results[{index}] != ({value}).ty() {{ return Err(self.constraint_error(_inst, {:?})); }}", format!("result {index} must have the type of `{name}`")).unwrap();
            storage_used = true;
        }
    }
    if op.signature_source.is_some() {
        use crate::model::SignatureSource;
        let error = "self.constraint_error(_inst, \"missing function or signature\")";
        body.push_str(&emit_signature(
            op,
            &emitter.projections,
            |source, value| {
                let id = match source {
                    SignatureSource::Function(_) => {
                        format!("_module.functions.get({value}).ok_or_else(|| {error})?.signature")
                    }
                    SignatureSource::Signature(_) => value.into(),
                    SignatureSource::Value(_) => format!(
                        "_dfg.value_type({value}).as_callable().ok_or_else(|| {error})?.0"
                    ),
                };
                format!("{{ let signature = _module.signatures().get({id}).ok_or_else(|| {error})?; (signature.params(), signature.returns()) }}")
            },
            |role, values, types| {
                format!(
                    "self.validate_values({:?}, {role:?}, {values}, {types}.iter().copied())?;",
                    op.mnemonic
                )
            },
            "_dfg.inst_results(_inst)",
        ));
        storage_used = true;
    }
    // table(cases, default) requires a default in its physical sequence.
    for (field, binding) in op.bindings() {
        if matches!(binding, crate::model::Binding::Table { .. }) {
            let index = format.fields.iter().position(|f| f.name == *field).unwrap();
            writeln!(body, "if _f{index}.is_empty() {{ return Err(self.constraint_error(_inst, \"branch table must contain a default destination\")); }}").unwrap();
            storage_used = true;
        }
    }
    body.push_str(&emit_checks(
        &op.constraints,
        &mut emitter,
        contexts,
        type_checks,
        error,
    ));
    storage_used |= emitter.storage_used.get();
    // Type-only predicates use already checked operand/result slices and
    // therefore work for alternate instruction layouts without reprojection.
    if storage_used {
        let fields = format
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| format!("{}: _f{i}", f.name))
            .collect::<Vec<_>>()
            .join(", ");
        let mismatch = if defs.storage.formats.len() + defs.storage.alternatives.len() == 1 {
            ""
        } else {
            " else { unreachable!(\"checked constraint storage\") }"
        };
        body.insert_str(
            0,
            &format!(
                "                let InstView::{} {{ {fields} }} = data{mismatch};\n",
                format.name
            ),
        );
    }
    body
}
