use super::{CheckedRule, PatternKind, Recipe, capture_code};
use std::{collections::BTreeMap, fmt::Write};

/// Generate direct operand/constant reductions and e-graph construction plans.
/// Structural matching belongs exclusively to the graph query compiler.
pub(in crate::rules) fn emit_local(
    defs: &crate::Definitions,
    rules: &[CheckedRule],
    opcode: &str,
    types: &str,
) -> String {
    let mut code = String::new();
    code.push_str("#[allow(unused_mut, unused_variables, unused_assignments)]\npub(crate) fn accepts(rule: usize, cx: &crate::rewrite::Context<'_, '_>, captures: &[crate::passes::expression::ExprValue], nodes: &[crate::passes::expression::ExprValue]) -> bool { (|| { match rule {\n");
    for (id, rule) in rules.iter().enumerate().filter(|(_, r)| !r.is_flat()) {
        writeln!(code, "{id} => {{").unwrap();
        emit_conditions(defs, rule, &mut code, types);
        code.push_str("Some(()) },\n");
    }
    code.push_str("_ => unreachable!(\"generated rule ID\"),\n} })().is_some() }\n");
    code.push_str("#[allow(unused_mut, unused_variables)]\npub(crate) fn plan(rule: usize, cx: &crate::rewrite::Context<'_, '_>, captures: &[crate::passes::expression::ExprValue], nodes: &[crate::passes::expression::ExprValue]) -> Option<crate::rewrite::Plan> { match rule {\n");
    for (id, rule) in rules.iter().enumerate().filter(|(_, r)| !r.is_flat()) {
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
    emit_flat(defs, rules, &mut code, opcode, types);
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
fn emit_flat(
    defs: &crate::Definitions,
    rules: &[CheckedRule],
    code: &mut String,
    opcode: &str,
    types: &str,
) {
    let mut groups = BTreeMap::<&str, Vec<&CheckedRule>>::new();
    for rule in rules.iter().filter(|r| r.is_flat()) {
        groups.entry(rule.opcode()).or_default().push(rule);
    }
    let ops = groups
        .keys()
        .map(|n| format!("{opcode}::{n}"))
        .collect::<Vec<_>>();
    let supported = if ops.is_empty() {
        "false".into()
    } else {
        format!("matches!(opcode, {})", ops.join(" | "))
    };
    writeln!(
        code,
        "pub(super) fn can_fold(opcode: {opcode}) -> bool {{ {supported} }}"
    )
    .unwrap();
    writeln!(code, "#[allow(unused_variables)]\npub(super) fn fold<V: Copy + Eq>(fields: &veloc_mir::InstFields, ty: {types}, args: &[V], mut constant: impl FnMut(V) -> Option<ScalarConst>) -> Option<crate::evaluate::Fold> {{ match fields.opcode() {{").unwrap();
    for (op, rules) in groups {
        writeln!(code, "{opcode}::{op} => {{").unwrap();
        for rule in rules {
            let PatternKind::Operation {
                args, attributes, ..
            } = &rule.pattern[0].kind
            else {
                unreachable!()
            };
            let orders = rule.orders(0);
            for order in orders {
                let column = |s: usize| order[args.iter().position(|&a| a == s).unwrap()];
                let mut checks = vec![
                    crate::types::generate::accepts(&rule.pattern[0].ty.domain, "ty"),
                    format!("args.len() == {}", args.len()),
                ];
                if !attributes.is_empty() {
                    let fields = attributes
                        .iter()
                        .enumerate()
                        .map(|(i, a)| (a.name.clone(), format!("attr_0_{i}")))
                        .collect();
                    let operation = defs.ops.iter().find(|o| o.name == op).unwrap();
                    code.push_str(&crate::storage::compact::bind_attributes(
                        operation,
                        &defs.storage,
                        &fields,
                        "fields",
                        "return None;",
                    ));
                    for (i, attribute) in attributes.iter().enumerate() {
                        let expected = attribute.value.code();
                        if expected != format!("attr_0_{i}") {
                            checks.push(format!("attr_0_{i} == {expected}"));
                        }
                    }
                }
                for &slot in args {
                    let c = column(slot);
                    match rule.pattern[slot].kind {
                        PatternKind::Value(previous) if previous != slot => checks.push(format!("crate::evaluate::same_value(args[{c}], args[{}], &mut constant)", column(previous))),
                        PatternKind::Constant(bits) => checks.push(format!("crate::evaluate::matches_constant(constant(args[{c}]), {bits}u64, true)")),
                        _ => {}
                    }
                }
                let result = match &rule.replacement {
                    Recipe::Value(slot) => {
                        format!("crate::evaluate::Fold::Operand({})", column(*slot))
                    }
                    Recipe::Constant(_, bits) => format!(
                        "crate::evaluate::Fold::Constant(ScalarConst::from_bits(ty, {bits}u64 & u64::MAX.checked_shr(64u32.checked_sub(ty.element_bits()?)?)?)?)"
                    ),
                    _ => unreachable!(),
                };
                writeln!(
                    code,
                    "if {} {{ return Some({result}); }}",
                    checks.join(" && ")
                )
                .unwrap();
            }
        }
        code.push_str("None\n},\n");
    }
    code.push_str("_ => None,\n} }\n");
}
