//! Compile independent result construction and type validation from signatures.
//! Bindings exist only here: runtime code refers directly to operand/result slots.
use crate::model::{Definitions, Pattern, TypeDef, TypeList};
use crate::types::generate::Sets;
use std::collections::BTreeMap;
use std::fmt::Write;

type Bindings = BTreeMap<u8, String>;

fn pattern(p: &Pattern, sets: &Sets) -> String {
    match p {
        Pattern::Property(name, _) => format!("type({name})"),
        Pattern::Callable => "Callable".into(),
        Pattern::Set(set) => sets.describe(set).into(),
        Pattern::Exact(ty) => ty.clone(),
        Pattern::Bind(var, set) => format!("T{var}: {}", sets.describe(set)),
        Pattern::Same(var) => format!("T{var}"),
        Pattern::ElementOf(var) => format!("element(T{var})"),
        Pattern::VectorOf(var) => format!("vector(T{var})"),
    }
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

fn check_list(
    out: &mut String,
    list: &TypeList,
    results: bool,
    bindings: &mut Bindings,
    sets: &Sets,
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
            Pattern::Callable => format!("veloc_types::traits::TypeInfo::is_callable({value})"),
            Pattern::Set(set) | Pattern::Property(_, set) => {
                format!("{}.accepts({value})", sets.reference(set))
            }
            Pattern::Exact(ty) => format!("{value} == {}", crate::types::rust_type(ty)),
            Pattern::Bind(var, set) => {
                let set = format!("{}.accepts({value})", sets.reference(set));
                if let Some(bound) = bindings.get(var) {
                    format!("{set} && {value} == {bound}")
                } else {
                    bindings.insert(*var, value.clone());
                    set
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
        };
        writeln!(out, "    if !({condition}) {{\n        return Err(super::TypeError::Pattern {{\n            results: {results}, index: {index}, expected: {:?}, got: {value},\n        }});\n    }}", pattern(p, sets)).unwrap();
    }
}

fn binding(bindings: &Bindings, var: u8) -> &str {
    bindings
        .get(&var)
        .expect("checked type variable is bound before use")
}

fn check_results(out: &mut String, ty: &TypeDef, bindings: &Bindings, sets: &Sets) {
    check_list(out, &ty.results, true, &mut bindings.clone(), sets);
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

fn emit_rule(id: usize, ty: &TypeDef, sets: &Sets, out: &mut String) {
    let mut operands = String::new();
    let mut bindings = Bindings::new();
    check_list(&mut operands, &ty.operands, false, &mut bindings, sets);
    let mut validate = operands;
    check_results(&mut validate, ty, &bindings, sets);
    validate.push_str("    Ok(())\n");
    function(out, &format!("validate_{id}"), ty, &validate);
}

pub(crate) fn generate_validation(defs: &Definitions, sets: &Sets, opcode: &str, out: &mut String) {
    validation_rules(defs, sets, opcode, out);
}

fn validation_rules<'a>(
    defs: &'a Definitions,
    sets: &Sets,
    opcode: &str,
    validation: &mut String,
) -> (BTreeMap<&'a TypeDef, usize>, Vec<Vec<&'a str>>) {
    let (ids, groups) = signature_groups(defs);
    for (ty, &id) in &ids {
        emit_rule(id, ty, sets, validation);
    }
    writeln!(validation, "impl {opcode} {{").unwrap();
    validation.push_str("/// Validate operand and result types without constructing an instruction.\n#[inline]\npub fn validate_types(self, operands: &[crate::Type], results: &[crate::Type]) -> core::result::Result<(), super::TypeError> {\n    match self {\n");
    for op in &defs.ops {
        let id = ids[&op.signature];
        writeln!(
            validation,
            "Self::{} => {{ validate_{id}(operands, results)?;",
            op.name
        )
        .unwrap();
        let mut emitter =
            crate::model::expr::Emitter::types(op, BTreeMap::new(), "operands", "results");
        for constraint in &op.constraints {
            if !constraint.type_only || constraint.redundant() {
                continue;
            }
            let error = format!("super::TypeError::Constraint({:?})", constraint.text);
            emitter.error = Some(error.clone());
            validation.push_str(&constraint.emit(&emitter, &format!("return Err({error})")));
        }
        validation.push_str("Ok(()) },\n");
    }
    validation.push_str("    }\n}\n}\n");

    (ids, groups)
}

fn signature_groups(defs: &Definitions) -> (BTreeMap<&TypeDef, usize>, Vec<Vec<&str>>) {
    let mut ids = BTreeMap::new();
    let mut groups: Vec<Vec<&str>> = Vec::new();
    for op in &defs.ops {
        let next = groups.len();
        let id = *ids.entry(&op.signature).or_insert(next);
        if id == next {
            groups.push(Vec::new());
        }
        groups[id].push(&op.name);
    }
    (ids, groups)
}
