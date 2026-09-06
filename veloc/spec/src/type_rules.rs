//! Compile type contracts into shared, straight-line checks and inference.
//! Bindings exist only here: runtime code refers directly to operand/result slots.
use crate::documentation::pattern;
use crate::model::{Definitions, Pattern, Relation, Slot, TypeDef, TypeList};
use crate::type_gen::Classes;
use std::collections::BTreeMap;
use std::fmt::Write;

type Bindings = BTreeMap<u8, String>;

pub(crate) fn generate(
    defs: &Definitions,
    classes: &Classes,
    types: &mut String,
    ops: &mut String,
) {
    let mut ids = BTreeMap::new();
    let mut groups: Vec<Vec<&str>> = Vec::new();
    for op in &defs.ops {
        let ty = &op.signature;
        // Structural equality preserves all checks and diagnostic positions.
        let next = groups.len();
        let id = *ids.entry(ty).or_insert(next);
        if id == next {
            emit_rule(id, ty, classes, types);
            groups.push(Vec::new());
        }
        groups[id].push(&op.name);
    }
    ops.push_str("impl Opcode {\n");
    for (method, extra, result, handler) in [
        (
            "validate_types",
            ", results: &[crate::Type]",
            "()",
            "validate",
        ),
        (
            "infer_result_types",
            "",
            "crate::opspec::ResultTypes",
            "infer",
        ),
    ] {
        writeln!(ops, "/// Execute the shared type contract generated from this opcode's definition.\n#[inline]\npub fn {method}(self, operands: &[crate::Type]{extra}) -> core::result::Result<{result}, crate::opspec::TypeError> {{\n    match self {{").unwrap();
        for (id, names) in groups.iter().enumerate() {
            let arms = names
                .iter()
                .map(|name| format!("Self::{name}"))
                .collect::<Vec<_>>()
                .join(" | ");
            let args = if handler == "validate" {
                "operands, results"
            } else {
                "operands"
            };
            writeln!(
                ops,
                "        {arms} => crate::opspec::type_rules::{handler}_{id}({args}),"
            )
            .unwrap();
        }
        ops.push_str("    }\n}\n");
    }
    // Text may reference values defined in later blocks. Only the contracts
    // whose results are independent of an unresolved variadic prefix may defer.
    ops.push_str("pub(crate) fn infer_text_result_types(self, operands: &[crate::Type]) -> core::result::Result<(crate::opspec::ResultTypes, bool), crate::opspec::TypeError> {\nlet inferred = self.infer_result_types(operands);\nmatch self {\n");
    for (signature, id) in &ids {
        if !matches!(&signature.operands, TypeList::Variadic(prefix) if !prefix.is_empty()) {
            continue;
        }
        let result = match &signature.results {
            TypeList::Signature => "crate::opspec::ResultTypes::Signature",
            TypeList::Fixed(patterns) if patterns.is_empty() => {
                "crate::opspec::ResultTypes::Inferred(Default::default())"
            }
            _ => continue,
        };
        let arms = groups[*id]
            .iter()
            .map(|name| format!("Self::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(ops, "{arms} => match inferred {{\nErr(crate::opspec::TypeError::Pattern {{ results: false, got: crate::Type::INVALID, .. }}) => Ok(({result}, true)),\nother => other.map(|types| (types, false)),\n}},").unwrap();
    }
    ops.push_str("_ => inferred.map(|types| (types, false)),\n}\n}\n}\n");
}

fn slot(slot: Slot) -> String {
    format!(
        "{}[{}]",
        if slot.result { "results" } else { "operands" },
        slot.index
    )
}

fn relation(r: &Relation) -> String {
    let (lhs, rhs) = (slot(r.lhs), slot(r.rhs));
    match r.kind.as_str() {
        "wider" => format!("{rhs} must have more bits per lane than {lhs}"),
        "narrower" => format!("{rhs} must have fewer bits per lane than {lhs}"),
        "same_width_distinct" => {
            format!("{lhs} and {rhs} must be distinct types with equal whole-value bit sizes")
        }
        _ => unreachable!("checked type relation"),
    }
}

fn check_list(
    out: &mut String,
    list: &TypeList,
    results: bool,
    bindings: &mut Bindings,
    classes: &Classes,
) {
    let (patterns, cmp) = match list {
        TypeList::Fixed(patterns) => (patterns, "!="),
        TypeList::Variadic(prefix) if !prefix.is_empty() => (prefix, "<"),
        TypeList::Variadic(_) | TypeList::Signature => return,
    };
    let values = if results { "results" } else { "operands" };
    let arity = match (cmp, patterns.len()) {
        ("!=", 0) => format!("!{values}.is_empty()"),
        ("<", 1) => format!("{values}.is_empty()"),
        (_, len) => format!("{values}.len() {cmp} {len}"),
    };
    writeln!(out, "    if {arity} {{\n        return Err(super::TypeError::Arity {{ results: {results}, expected: {}, got: {values}.len() }});\n    }}", patterns.len()).unwrap();
    for (index, p) in patterns.iter().enumerate() {
        let value = format!("{values}[{index}]");
        let condition = match p {
            Pattern::Class(class) => format!("{}.accepts({value})", classes.reference(class)),
            Pattern::Exact(ty) => format!("{value} == Type::{ty}"),
            Pattern::Bind(var, class) => {
                let class = format!("{}.accepts({value})", classes.reference(class));
                if let Some(bound) = bindings.get(var) {
                    format!("{class} && {value} == {bound}")
                } else {
                    bindings.insert(*var, value.clone());
                    class
                }
            }
            Pattern::Same(var) => format!("{value} == {}", binding(bindings, *var)),
            Pattern::ElementOf(var) => format!(
                "{}.as_vector().is_some_and(|vector| {value} == vector.element_type().as_type())",
                binding(bindings, *var)
            ),
            Pattern::VectorOf(var) => format!(
                "{}.as_scalar().is_some_and(|scalar| {value}.as_vector().is_some_and(|vector| vector.element_type() == scalar))",
                binding(bindings, *var)
            ),
            Pattern::ShapeOf(var, class) => format!(
                "{}.accepts({value}) && super::same_shape({}, {value})",
                classes.reference(class),
                binding(bindings, *var)
            ),
        };
        writeln!(out, "    if !({condition}) {{\n        return Err(super::TypeError::Pattern {{\n            results: {results}, index: {index}, expected: {:?}, got: {value},\n        }});\n    }}", pattern(p, classes)).unwrap();
    }
}

fn binding(bindings: &Bindings, var: u8) -> &str {
    bindings
        .get(&var)
        .expect("checked type variable is bound before use")
}

fn check_results(
    out: &mut String,
    ty: &TypeDef,
    bindings: &Bindings,
    classes: &Classes,
    inferred: bool,
) {
    if inferred {
        // Result count and Same/Exact/ElementOf checks follow directly from
        // the expressions we emitted. Only a Bind's class can add a constraint.
        for (index, p) in ty
            .results
            .patterns()
            .expect("inferred fixed results")
            .iter()
            .enumerate()
        {
            if let Pattern::Bind(_, class) = p {
                writeln!(out, "    if !{}.accepts(results[{index}]) {{\n        return Err(super::TypeError::Pattern {{ results: true, index: {index}, expected: {:?}, got: results[{index}] }});\n    }}", classes.reference(class), pattern(p, classes)).unwrap();
            }
        }
    } else {
        check_list(out, &ty.results, true, &mut bindings.clone(), classes);
    }
    for r in &ty.relations {
        let (lhs, rhs) = (slot(r.lhs), slot(r.rhs));
        let condition = match r.kind.as_str() {
            "wider" => format!(
                "{lhs}.element_bits().zip({rhs}.element_bits()).is_some_and(|(from, to)| to > from)"
            ),
            "narrower" => format!(
                "{lhs}.element_bits().zip({rhs}.element_bits()).is_some_and(|(from, to)| to < from)"
            ),
            "same_width_distinct" => format!(
                "{lhs}.bit_size().zip({rhs}.bit_size()).is_some_and(|(a, b)| {lhs} != {rhs} && a == b)"
            ),
            _ => unreachable!("checked type relation"),
        };
        writeln!(out, "    if !({condition}) {{\n        return Err(super::TypeError::Relation({:?}));\n    }}", relation(r)).unwrap();
    }
}

fn function(out: &mut String, name: &str, ty: &TypeDef, results: bool, body: &str) {
    // Empty/variadic contracts may not inspect a parameter at all.
    let inspected = |list: &TypeList| match list {
        TypeList::Fixed(_) => true,
        TypeList::Variadic(prefix) => !prefix.is_empty(),
        TypeList::Signature => false,
    };
    let arg = if inspected(&ty.operands) {
        "operands"
    } else {
        "_operands"
    };
    let extra = if results { ", results: &[Type]" } else { "" };
    let extra = if results && !inspected(&ty.results) {
        ", _results: &[Type]"
    } else {
        extra
    };
    let result = if results { "()" } else { "super::ResultTypes" };
    writeln!(out, "#[inline]\npub(crate) fn {name}({arg}: &[Type]{extra}) -> core::result::Result<{result}, super::TypeError> {{\n{body}}}\n").unwrap();
}

fn emit_rule(id: usize, ty: &TypeDef, classes: &Classes, out: &mut String) {
    let mut operands = String::new();
    let mut bindings = Bindings::new();
    check_list(&mut operands, &ty.operands, false, &mut bindings, classes);
    let mut validate = operands.clone();
    check_results(&mut validate, ty, &bindings, classes, false);
    validate.push_str("    Ok(())\n");
    function(out, &format!("validate_{id}"), ty, true, &validate);

    let mut infer = operands;
    match &ty.results {
        TypeList::Signature => infer.push_str("    Ok(super::ResultTypes::Signature)\n"),
        TypeList::Variadic(_) => infer.push_str("    Ok(super::ResultTypes::Explicit)\n"),
        TypeList::Fixed(patterns) => {
            let expressions = patterns.iter().map(|p| match p {
                Pattern::Exact(ty) => Some(format!("Type::{ty}")),
                Pattern::Same(var) | Pattern::Bind(var, _) => bindings.get(var).cloned(),
                Pattern::ElementOf(var) => bindings.get(var).map(|bound| format!("match {bound}.as_vector() {{ Some(vector) => vector.element_type().as_type(), None => return Ok(super::ResultTypes::Explicit) }}")),
                Pattern::Class(_) | Pattern::VectorOf(_) | Pattern::ShapeOf(_, _) => None,
            }).collect::<Option<Vec<_>>>();
            if let Some(expressions) = expressions {
                writeln!(infer, "    let results = [{}];", expressions.join(", ")).unwrap();
                check_results(&mut infer, ty, &bindings, classes, true);
                infer.push_str("    Ok(super::ResultTypes::Inferred(smallvec::SmallVec::from_slice(&results)))\n");
            } else {
                infer.push_str("    Ok(super::ResultTypes::Explicit)\n");
            }
        }
    }
    function(out, &format!("infer_{id}"), ty, false, &infer);
}
