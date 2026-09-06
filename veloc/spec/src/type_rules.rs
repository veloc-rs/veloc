//! Compile independent result construction and type validation from signatures.
//! Bindings exist only here: runtime code refers directly to operand/result slots.
use crate::model::{Definitions, Pattern, Relation, Slot, TypeDef, TypeList};
use crate::type_gen::Classes;
use std::collections::BTreeMap;
use std::fmt::Write;

type Bindings = BTreeMap<u8, String>;

fn pattern(p: &Pattern, classes: &Classes) -> String {
    match p {
        Pattern::Class(class) => classes.describe(class).into(),
        Pattern::Exact(ty) => ty.clone(),
        Pattern::Bind(var, class) => format!("T{var}: {}", classes.describe(class)),
        Pattern::Same(var) => format!("T{var}"),
        Pattern::ElementOf(var) => format!("element(T{var})"),
        Pattern::VectorOf(var) => format!("vector(T{var})"),
        Pattern::ShapeOf(var, class) => format!("shape(T{var}, {})", classes.describe(class)),
    }
}

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
    ops.push_str("/// Validate operand and result types without constructing an instruction.\n#[inline]\npub fn validate_types(self, operands: &[crate::Type], results: &[crate::Type]) -> core::result::Result<(), crate::opspec::TypeError> {\n    match self {\n");
    for (id, names) in groups.iter().enumerate() {
        let arms = names
            .iter()
            .map(|name| format!("Self::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            ops,
            "        {arms} => crate::opspec::type_rules::validate_{id}(operands, results),"
        )
        .unwrap();
    }
    ops.push_str("    }\n}\n}\n");
    // Only the dynamic construction path needs opcode dispatch. Generated
    // builders use the same result expressions directly on their arguments.
    ops.push_str("impl crate::InstructionData {\n/// Determine result types without validating the instruction's type contract.\n/// Explicit types are used only when the signature cannot infer its results.\n/// Referenced values and physical storage must exist.\npub fn result_types(&self, dfg: &crate::dfg::DataFlowGraph, module: &crate::ModuleData, explicit: &[crate::Type]) -> core::result::Result<smallvec::SmallVec<[crate::Type; 2]>, &'static str> {\nuse crate::Type;\nlet _ = (dfg, module, explicit);\nmatch self.opcode() {\n");
    for (signature, id) in &ids {
        let arms = groups[*id]
            .iter()
            .map(|name| format!("crate::Opcode::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(ops, "{arms} => {{").unwrap();
        if matches!(signature.results, TypeList::Signature) {
            ops.push_str("let sig = self.call_info().expect(\"signature results require call metadata\").signature.resolve(module).ok_or(\"unknown function or signature\")?;\nlet sig = module.signatures.get(sig).ok_or(\"unknown signature\")?;\nOk(smallvec::SmallVec::from_slice(&sig.returns))\n");
        } else if let Some(results) = result_exprs(signature) {
            if results.iter().any(|r| !matches!(r, ResultExpr::Exact(_))) {
                ops.push_str("let mut operands = smallvec::SmallVec::<[Type; 4]>::new();\nself.visit_type_operands(dfg, |value| operands.push(dfg.value_type(value)));\n");
            }
            let operand = |index: usize| {
                if index == 0 {
                    "operands.first()".to_owned()
                } else {
                    format!("operands.get({index})")
                }
            };
            let expressions = results.iter().map(|r| match r {
                ResultExpr::Exact(ty) => format!("Type::{ty}"),
                ResultExpr::Operand(index) => format!("*{}.filter(|ty| ty.is_valid()).ok_or(\"result type requires a known operand type\")?", operand(*index)),
                ResultExpr::Element(index) => format!("{}.and_then(|ty| ty.as_vector()).ok_or(\"result element type requires a known vector operand\")?.element_type().as_type()", operand(*index)),
            }).collect::<Vec<_>>().join(", ");
            writeln!(ops, "Ok(smallvec::smallvec![{expressions}])").unwrap();
        } else {
            ops.push_str("if explicit.is_empty() { return Err(\"requires an explicit result type\"); }\nOk(smallvec::SmallVec::from_slice(explicit))\n");
        }
        ops.push_str("},\n");
    }
    ops.push_str("}\n}\n}\n");
}

/// Build-time expressions, never emitted as runtime descriptors.
pub(crate) enum ResultExpr {
    Exact(String),
    Operand(usize),
    Element(usize),
}

pub(crate) fn result_exprs(ty: &TypeDef) -> Option<Vec<ResultExpr>> {
    let mut bindings = BTreeMap::new();
    for (index, p) in ty
        .operands
        .patterns()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        if let Pattern::Bind(var, _) = p {
            bindings.entry(*var).or_insert(index);
        }
    }
    ty.results
        .patterns()?
        .iter()
        .map(|p| match p {
            Pattern::Exact(ty) => Some(ResultExpr::Exact(ty.clone())),
            Pattern::Same(var) | Pattern::Bind(var, _) => {
                bindings.get(var).copied().map(ResultExpr::Operand)
            }
            Pattern::ElementOf(var) => bindings.get(var).copied().map(ResultExpr::Element),
            _ => None,
        })
        .collect()
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

fn check_results(out: &mut String, ty: &TypeDef, bindings: &Bindings, classes: &Classes) {
    check_list(out, &ty.results, true, &mut bindings.clone(), classes);
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

fn function(out: &mut String, name: &str, ty: &TypeDef, body: &str) {
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
    let extra = if !inspected(&ty.results) {
        ", _results: &[Type]"
    } else {
        ", results: &[Type]"
    };
    writeln!(out, "#[inline]\npub(crate) fn {name}({arg}: &[Type]{extra}) -> core::result::Result<(), super::TypeError> {{\n{body}}}\n").unwrap();
}

fn emit_rule(id: usize, ty: &TypeDef, classes: &Classes, out: &mut String) {
    let mut operands = String::new();
    let mut bindings = Bindings::new();
    check_list(&mut operands, &ty.operands, false, &mut bindings, classes);
    let mut validate = operands;
    check_results(&mut validate, ty, &bindings, classes);
    validate.push_str("    Ok(())\n");
    function(out, &format!("validate_{id}"), ty, &validate);
}
