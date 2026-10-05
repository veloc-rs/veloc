use super::{CheckedRule, PatternKind, Recipe};
use std::{collections::BTreeMap, fmt::Write};

/// Generate the SSA matcher and construction plans from the same checked rules
/// consumed by the graph query compiler. Plans never perform structural lookup.
pub(in crate::rules) fn emit_local(rules: &[CheckedRule], opcode: &str, types: &str) -> String {
    let mut code = String::new();
    let mut groups = BTreeMap::<&str, Vec<usize>>::new();
    for (id, rule) in rules.iter().enumerate() {
        groups.entry(rule.opcode()).or_default().push(id);
    }
    let ops = groups
        .keys()
        .map(|n| format!("{opcode}::{n}"))
        .collect::<Vec<_>>();
    let supported = if ops.is_empty() {
        "false".into()
    } else {
        format!("matches!(op, {})", ops.join(" | "))
    };
    writeln!(
        code,
        "pub(crate) fn can_rewrite(op: {opcode}) -> bool {{ {supported} }}"
    )
    .unwrap();
    code.push_str("#[allow(unused_mut, unused_variables, unused_assignments)]\npub(crate) fn accepts<R: crate::rewrite::View + ?Sized>(rule: usize, cx: &crate::rewrite::Context<'_, R>, captures: &[R::Value], nodes: &[R::Value]) -> bool { (|| { match rule {\n");
    for (id, rule) in rules.iter().enumerate() {
        writeln!(code, "{id} => {{").unwrap();
        emit_conditions(rule, &mut code, types);
        code.push_str("Some(()) },\n");
    }
    code.push_str("_ => unreachable!(\"generated rule ID\"),\n} })().is_some() }\n");
    code.push_str("#[allow(unused_mut, unused_variables)]\npub(crate) fn plan<R: crate::rewrite::View + ?Sized>(rule: usize, cx: &crate::rewrite::Context<'_, R>, captures: &[R::Value], nodes: &[R::Value]) -> Option<crate::rewrite::Plan<R::Value>> { match rule {\n");
    for (id, rule) in rules.iter().enumerate() {
        writeln!(code, "{id} => {{").unwrap();
        emit_conditions(rule, &mut code, types);
        code.push_str("let mut plan = crate::rewrite::PlanBuilder::default();\n");
        let result = emit_recipe(&rule.replacement, &mut code, &mut 0, opcode, types);
        writeln!(
            code,
            "let result = {result};\nplan.finish(cx, result, cx.ty(values[0]))\n}},"
        )
        .unwrap();
    }
    code.push_str("_ => unreachable!(\"generated rule ID\"),\n} }\n");
    code.push_str("#[allow(unused_mut, unused_variables)]\npub(crate) fn rewrite(cx: &crate::rewrite::Context<'_>, root: Value) -> Option<crate::rewrite::Plan> { match cx.node(root)?.opcode {\n");
    for (op, ids) in &groups {
        writeln!(code, "{opcode}::{op} => {{").unwrap();
        for &id in ids {
            let rule = &rules[id];
            if rule.is_flat() {
                continue;
            }
            writeln!(
                code,
                "if let Some(plan) = (|| {{ let mut values = [root; {}]; let mut nodes = values;",
                rule.pattern.len()
            )
            .unwrap();
            emit_match(rule, id, 0, &mut code, opcode, types);
            code.push_str("})() { return Some(plan); }\n");
        }
        code.push_str("None\n},\n");
    }
    code.push_str("_ => None,\n} }\n");
    emit_flat(rules, &mut code, opcode, types);
    code
}
/// Conditions are shared by query filtering and checked construction. Query
/// filtering does not allocate a replacement plan; application checks again
/// after canonicalizing captured classes against the latest graph.
fn emit_conditions(rule: &CheckedRule, code: &mut String, types: &str) {
    writeln!(
        code,
        "let mut values = [captures[0]; {}];",
        rule.pattern.len()
    )
    .unwrap();
    for (i, slot) in rule.captures().iter().enumerate() {
        writeln!(code, "values[{slot}] = captures[{i}];").unwrap();
    }
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
            .map(|(j, a)| format!("{}: attr_{slot}_{j}", a.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(code, "let Properties::{op} {{ {bindings} }} = cx.properties(nodes[{i}])? else {{ return None; }};").unwrap();
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
        writeln!(code, "if !({}) {{ return None; }}", guard.code(types)).unwrap();
    }
}
fn emit_recipe(
    recipe: &Recipe,
    code: &mut String,
    next: &mut usize,
    opcode: &str,
    types: &str,
) -> String {
    match recipe {
        Recipe::Value(slot) => format!("crate::rewrite::Input::Value(values[{slot}])"),
        Recipe::Constant(ty, bits) => format!("plan.constant({}, {bits}u64)?", ty.code(types)),
        Recipe::Build {
            opcode: op,
            ty,
            args,
            attributes,
        } => {
            let mut inputs = Vec::new();
            for arg in args {
                let expr = emit_recipe(arg, code, next, opcode, types);
                let name = format!("r{next}");
                *next += 1;
                writeln!(code, "let {name} = {expr};").unwrap();
                inputs.push(name);
            }
            let properties = if attributes.is_empty() {
                "Properties::None".into()
            } else {
                format!(
                    "Properties::{op} {{ {} }}",
                    attributes
                        .iter()
                        .map(|a| format!("{}: {}", a.name, a.value.code()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            format!(
                "plan.build(cx, {opcode}::{op}, {}, &[{}], {properties})?",
                ty.code(types),
                inputs.join(", ")
            )
        }
    }
}
fn emit_match(
    rule: &CheckedRule,
    id: usize,
    slot: usize,
    code: &mut String,
    opcode: &str,
    types: &str,
) {
    if slot == rule.pattern.len() {
        for (slot, pattern) in rule.pattern.iter().enumerate() {
            let ty = format!("cx.ty(values[{slot}])");
            writeln!(
                code,
                "if !({}) || {ty} != {} {{ return None; }}",
                crate::types::generate::accepts(&pattern.ty.domain, &ty),
                pattern.ty.code(types)
            )
            .unwrap();
            match pattern.kind {
                PatternKind::Value(previous) if previous != slot => {
                    writeln!(
                        code,
                        "if values[{previous}] != values[{slot}] {{ return None; }}"
                    )
                    .unwrap();
                }
                PatternKind::Constant(bits) => {
                    writeln!(
                        code,
                        "if !cx.matches_constant(values[{slot}], {bits}u64) {{ return None; }}"
                    )
                    .unwrap();
                }
                _ => {}
            }
        }
        let captures = rule
            .captures()
            .iter()
            .map(|s| format!("values[{s}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let nodes = rule
            .nodes()
            .iter()
            .map(|s| format!("nodes[{s}]"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(code, "let plan = plan({id}, cx, &[{captures}], &[{nodes}])?;\nplan.profitable(cx.body, root, {}).then_some(plan)", rule.canonical).unwrap();
        return;
    }
    match &rule.pattern[slot].kind {
        PatternKind::Operation {
            opcode: op, args, ..
        } => {
            if rule.nodes().contains(&slot) {
                writeln!(code, "nodes[{slot}] = values[{slot}];").unwrap();
            }
            writeln!(code, "let node = cx.node(values[{slot}])?;\nif node.opcode != {opcode}::{op} {{ return None; }}").unwrap();
            let orders = rule.orders(slot);
            // Each continuation owns its failure: later operand permutations
            // and later rules still run when a nested match or policy rejects it.
            writeln!(
                code,
                "for order in {orders:?} {{\nif let Some(plan) = (|| {{"
            )
            .unwrap();
            for (i, arg) in args.iter().enumerate() {
                writeln!(code, "values[{arg}] = node.args[order[{i}]];").unwrap();
            }
            emit_match(rule, id, slot + 1, code, opcode, types);
            code.push_str("})() { return Some(plan); }\n}\nNone\n");
        }
        _ => emit_match(rule, id, slot + 1, code, opcode, types),
    }
}
fn emit_flat(rules: &[CheckedRule], code: &mut String, opcode: &str, types: &str) {
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
    writeln!(code, "#[allow(unused_variables)]\npub(super) fn fold<V: Copy + Eq>(opcode: {opcode}, ty: {types}, args: &[V], mut constant: impl FnMut(V) -> Option<ScalarConst>) -> Option<crate::evaluate::Fold> {{ match opcode {{").unwrap();
    for (op, rules) in groups {
        writeln!(code, "{opcode}::{op} => {{").unwrap();
        for rule in rules {
            let PatternKind::Operation { args, .. } = &rule.pattern[0].kind else {
                unreachable!()
            };
            let orders = rule.orders(0);
            for order in orders {
                let column = |s: usize| order[args.iter().position(|&a| a == s).unwrap()];
                let mut checks = vec![
                    crate::types::generate::accepts(&rule.pattern[0].ty.domain, "ty"),
                    format!("args.len() == {}", args.len()),
                ];
                for &slot in args {
                    let c = column(slot);
                    match rule.pattern[slot].kind {
                        PatternKind::Value(previous) if previous != slot => checks.push(format!("args[{c}] == args[{}]", column(previous))),
                        PatternKind::Constant(bits) => checks.push(format!("constant(args[{c}]).is_some_and(|c| c.ty().element_bits().and_then(|b| 64u32.checked_sub(b)).and_then(|s| u64::MAX.checked_shr(s)).is_some_and(|mask| c.to_bits() == ({bits}u64 & mask)))")),
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
