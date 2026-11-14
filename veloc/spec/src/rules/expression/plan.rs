use super::{CheckedRule, PatternKind, Recipe, capture_code};
use std::fmt::Write;

/// Generate guards and construction recipes alongside the e-class queries.
pub(in crate::rules) fn emit_plans(
    defs: &crate::Definitions,
    rules: &[CheckedRule],
    opcode: &str,
    types: &str,
) -> String {
    let mut code = String::new();
    code.push_str("#[allow(unused_mut, unused_variables, unused_assignments)]\nfn accepts(rule: usize, cx: &crate::rewrite::Context<'_, '_>, captures: &[crate::passes::expression::ExprValue], nodes: &[crate::passes::expression::ExprValue]) -> bool { (|| { match rule {\n");
    for (id, rule) in rules
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.can_fold_locally())
    {
        writeln!(code, "{id} => {{").unwrap();
        emit_conditions(defs, rule, &mut code, types);
        code.push_str("Some(()) },\n");
    }
    code.push_str("_ => unreachable!(\"generated rule ID\"),\n} })().is_some() }\n");
    code.push_str("#[allow(unused_mut, unused_variables)]\nfn plan(rule: usize, cx: &crate::rewrite::Context<'_, '_>, captures: &[crate::passes::expression::ExprValue], nodes: &[crate::passes::expression::ExprValue]) -> Option<crate::rewrite::Plan> { match rule {\n");
    for (id, rule) in rules
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.can_fold_locally())
    {
        writeln!(code, "{id} => {{").unwrap();
        emit_conditions(defs, rule, &mut code, types);
        code.push_str("let mut plan = crate::rewrite::PlanBuilder::default();\n");
        let captures = rule.captures();
        let result = emit_recipe(
            defs,
            &rule.replacement,
            &mut code,
            &mut 0,
            opcode,
            types,
            &captures,
        );
        let root = capture_code(&captures, 0);
        writeln!(
            code,
            "let result = {result};\nplan.finish(cx, result, cx.ty({root}))\n}},"
        )
        .unwrap();
    }
    code.push_str("_ => unreachable!(\"generated rule ID\"),\n} }\n");
    code
}
/// Conditions are shared by query filtering and checked construction. Query
/// filtering does not allocate a replacement plan; application checks again
/// after canonicalizing captured classes against the latest graph.
fn emit_conditions(defs: &crate::Definitions, rule: &CheckedRule, code: &mut String, types: &str) {
    for (i, slot) in rule.nodes().into_iter().enumerate() {
        let PatternKind::Operation {
            opcode: op,
            attributes,
            ..
        } = &rule.pattern[slot].kind
        else {
            unreachable!()
        };
        let bindings = attributes
            .iter()
            .enumerate()
            .map(|(j, a)| (a.name.clone(), format!("attr_{slot}_{j}")))
            .collect();
        let operation = defs.ops.iter().find(|o| o.name == *op).unwrap();
        code.push_str(&crate::storage::compact::bind_attributes(
            operation,
            &defs.storage,
            &bindings,
            &format!("cx.fields(nodes[{i}])?"),
            "return None;",
        ));
    }
    for slot in rule.nodes() {
        let PatternKind::Operation { attributes, .. } = &rule.pattern[slot].kind else {
            unreachable!()
        };
        for (j, attribute) in attributes.iter().enumerate() {
            let expected = attribute.value.code();
            if expected != format!("attr_{slot}_{j}") {
                writeln!(code, "if attr_{slot}_{j} != {expected} {{ return None; }}").unwrap();
            }
        }
    }
    if let Some(guard) = &rule.guard {
        writeln!(
            code,
            "if !({}) {{ return None; }}",
            guard.code(types, &rule.captures())
        )
        .unwrap();
    }
}
fn emit_recipe(
    defs: &crate::Definitions,
    recipe: &Recipe,
    code: &mut String,
    next: &mut usize,
    opcode: &str,
    types: &str,
    captures: &[usize],
) -> String {
    match recipe {
        Recipe::Value(slot) => format!(
            "crate::rewrite::Input::Value({})",
            capture_code(captures, *slot)
        ),
        Recipe::Constant(ty, bits) => {
            format!("plan.constant({}, {bits}u64)?", ty.code(types, captures))
        }
        Recipe::Build {
            opcode: op,
            ty,
            args,
            attributes,
        } => {
            let mut inputs = Vec::new();
            for arg in args {
                let expr = emit_recipe(defs, arg, code, next, opcode, types, captures);
                let name = format!("r{next}");
                *next += 1;
                writeln!(code, "let {name} = {expr};").unwrap();
                inputs.push(name);
            }
            let operation = defs.ops.iter().find(|o| o.name == *op).unwrap();
            let attributes = attributes
                .iter()
                .map(|a| (a.name.clone(), a.value.code()))
                .collect();
            let fields = crate::generate::fields::constructor(
                defs,
                operation,
                &attributes,
                "veloc_mir::InstFields",
                &format!("{opcode}::{op}"),
            );
            format!(
                "plan.build(cx, {fields}, {}, &[{}])?",
                ty.code(types, captures),
                inputs.join(", ")
            )
        }
    }
}
