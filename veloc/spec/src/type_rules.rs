//! Compile independent result construction and type validation from signatures.
//! Bindings exist only here: runtime code refers directly to operand/result slots.
use crate::model::{
    Binding, Definitions, Op, Pattern, Relation, SignatureSource, Slot, TypeDef, TypeList,
};
use crate::type_gen::Classes;
use std::collections::BTreeMap;
use std::fmt::Write;

type Bindings = BTreeMap<u8, String>;

fn pattern(p: &Pattern, classes: &Classes) -> String {
    match p {
        Pattern::Property(name, _) => format!("type({name})"),
        Pattern::Callable => "Callable".into(),
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
    validation: &mut String,
    instructions: &mut String,
) {
    let (ids, groups) = validation_rules(defs, classes, "crate::Opcode", validation);
    // Only the dynamic construction path needs opcode dispatch. Generated
    // builders use the same result expressions directly on their arguments.
    instructions.push_str("impl crate::InstructionView<'_> {\n/// Determine result types without validating the instruction's type contract.\n/// Explicit types are used only when the signature cannot infer its results.\n/// Referenced values and physical storage must exist.\npub fn result_types(&self, dfg: &crate::dfg::DataFlowGraph, module: &crate::ModuleData, explicit: &[crate::Type]) -> core::result::Result<smallvec::SmallVec<[crate::Type; 2]>, &'static str> {\nuse crate::Type;\nlet _ = (dfg, module, explicit);\nmatch (self.opcode(), self) {\n");
    for (signature, id) in &ids {
        if matches!(signature.results, TypeList::Signature) {
            // Equal type schemes can still resolve their signatures differently.
            // Specialize each source directly instead of emitting a runtime tag.
            for op in defs.ops.iter().filter(|op| &op.signature == *signature) {
                signature_results(op, instructions);
            }
            continue;
        }
        if signature
            .results
            .patterns()
            .is_some_and(|p| p.iter().any(|p| matches!(p, Pattern::Property(..))))
        {
            for op in defs.ops.iter().filter(|op| &op.signature == *signature) {
                let format = defs
                    .storage
                    .formats
                    .iter()
                    .find(|f| f.name == op.format)
                    .unwrap();
                let fields = format
                    .fields
                    .iter()
                    .map(|f| format!("{}: _{}", f.name, f.name))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(
                    instructions,
                    "(crate::Opcode::{}, Self::{} {{ {fields} }}) => {{",
                    op.name, op.format
                )
                .unwrap();
                let projections = crate::packing::projections(
                    op,
                    format,
                    "dfg",
                    |f| format!("*_{f}"),
                    |v| format!("{v}.ok_or(\"missing property storage\")?"),
                );
                emit_results(signature, instructions, |name| {
                    let value = &projections.iter().find(|(p, _)| p == name).unwrap().1;
                    format!("({value}).ty()")
                });
                instructions.push_str("},\n");
            }
            continue;
        }
        let arms = groups[*id]
            .iter()
            .map(|name| format!("crate::Opcode::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(instructions, "({arms}, _) => {{").unwrap();
        emit_results(signature, instructions, |_| {
            unreachable!("property results emitted per operation")
        });
        instructions.push_str("},\n");
    }
    if defs.storage.formats.len() > 1
        && defs.ops.iter().any(|op| {
            op.signature_source.is_some()
                || op
                    .signature
                    .results
                    .patterns()
                    .is_some_and(|p| p.iter().any(|p| matches!(p, Pattern::Property(..))))
        })
    {
        instructions.push_str(
            "_ => Err(\"result type source is stored in an incompatible instruction format\"),\n",
        );
    }
    instructions.push_str("}\n}\n}\n");
}

fn emit_results(signature: &TypeDef, instructions: &mut String, property: impl Fn(&str) -> String) {
    if let Some(results) = result_exprs(signature) {
        if results
            .iter()
            .any(|r| matches!(r, ResultExpr::Operand(_) | ResultExpr::Element(_)))
        {
            instructions.push_str("let mut operands = smallvec::SmallVec::<[Type; 4]>::new();\nself.visit_type_operands(|value| operands.push(dfg.value_type(value)));\n");
        }
        let operand = |index: usize| {
            if index == 0 {
                "operands.first()".to_owned()
            } else {
                format!("operands.get({index})")
            }
        };
        let expressions = results.iter().map(|r| match r {
                ResultExpr::Property(name) => property(name),
                ResultExpr::Exact(ty) => format!("Type::{ty}"),
                ResultExpr::Operand(index) => format!("*{}.filter(|ty| ty.is_valid()).ok_or(\"result type requires a known operand type\")?", operand(*index)),
                ResultExpr::Element(index) => format!("{}.and_then(|ty| ty.as_vector()).ok_or(\"result element type requires a known vector operand\")?.element_type().as_type()", operand(*index)),
            }).collect::<Vec<_>>().join(", ");
        writeln!(instructions, "Ok(smallvec::smallvec![{expressions}])").unwrap();
    } else {
        instructions.push_str("if explicit.is_empty() { return Err(\"requires an explicit result type\"); }\nOk(smallvec::SmallVec::from_slice(explicit))\n");
    }
}

fn signature_results(op: &Op, out: &mut String) {
    let source = op
        .signature_source
        .as_ref()
        .expect("checked signature source");
    let (name, id) = match source {
        SignatureSource::Function(name) => (
            name,
            "module.functions.get(*source).ok_or(\"unknown function\")?.signature",
        ),
        SignatureSource::Signature(name) => (name, "*source"),
        SignatureSource::Value(name) => (
            name,
            "dfg.values().get(*source).and_then(|value| value.ty.as_callable()).ok_or(\"unknown or non-callable value\")?.0",
        ),
    };
    let field = op
        .bindings()
        .iter()
        .find_map(|(field, binding)| {
            matches!(binding, Binding::Name(param) if param == name).then_some(field)
        })
        .expect("checked signature source storage");
    writeln!(out, "(crate::Opcode::{}, Self::{} {{ {field}: source, .. }}) => {{\nlet sig = {id};\nlet sig = module.signatures.get(sig).ok_or(\"unknown signature\")?;\nOk(smallvec::SmallVec::from_slice(&sig.returns))\n}},", op.name, op.format).unwrap();
}

/// Build-time expressions, never emitted as runtime descriptors.
pub(crate) enum ResultExpr {
    Property(String),
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
            Pattern::Property(name, _) => Some(ResultExpr::Property(name.clone())),
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
            Pattern::Callable => format!("{value}.is_callable()"),
            Pattern::Class(class) | Pattern::Property(_, class) => {
                format!("{}.accepts({value})", classes.reference(class))
            }
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
                "{}.accepts({value}) && same_shape({}, {value})",
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
    writeln!(out, "#[inline]\nfn {name}({arg}: &[Type]{extra}) -> core::result::Result<(), super::TypeError> {{\n{body}}}\n").unwrap();
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

pub(crate) fn generate_validation(
    defs: &Definitions,
    classes: &Classes,
    opcode: &str,
    out: &mut String,
) {
    validation_rules(defs, classes, opcode, out);
}

fn validation_rules<'a>(
    defs: &'a Definitions,
    classes: &Classes,
    opcode: &str,
    validation: &mut String,
) -> (BTreeMap<&'a TypeDef, usize>, Vec<Vec<&'a str>>) {
    validation.push_str("#[allow(dead_code)]\nfn same_shape(bound: Type, ty: Type) -> bool {\nif let Some(bound) = bound.as_vector() {\nty.as_vector().is_some_and(|vector| vector.shape() == bound.shape())\n} else { ty.as_scalar().is_some() }\n}\n");
    let mut ids = BTreeMap::new();
    let mut groups: Vec<Vec<&str>> = Vec::new();
    for op in &defs.ops {
        let ty = &op.signature;
        // Structural equality preserves all checks and diagnostic positions.
        let next = groups.len();
        let id = *ids.entry(ty).or_insert(next);
        if id == next {
            emit_rule(id, ty, classes, validation);
            groups.push(Vec::new());
        }
        groups[id].push(&op.name);
    }
    writeln!(validation, "impl {opcode} {{").unwrap();
    validation.push_str("/// Validate operand and result types without constructing an instruction.\n#[inline]\npub fn validate_types(self, operands: &[crate::Type], results: &[crate::Type]) -> core::result::Result<(), super::TypeError> {\n    match self {\n");
    for (id, names) in groups.iter().enumerate() {
        let arms = names
            .iter()
            .map(|name| format!("Self::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            validation,
            "        {arms} => validate_{id}(operands, results),"
        )
        .unwrap();
    }
    validation.push_str("    }\n}\n}\n");

    (ids, groups)
}
