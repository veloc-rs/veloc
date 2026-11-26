//! Direct reductions use only the root, operand identities and constant facts.
use super::{CheckedRule, PatternKind, Recipe};
use std::{collections::BTreeMap, fmt::Write};

pub(in crate::rules) fn emit_folds(
    defs: &crate::Definitions,
    rules: &[CheckedRule],
    opcode: &str,
    types: &str,
) -> String {
    let mut code = String::new();
    let mut groups = BTreeMap::<&str, Vec<&CheckedRule>>::new();
    for rule in rules.iter().filter(|r| r.can_fold_locally()) {
        groups.entry(rule.opcode()).or_default().push(rule);
    }
    let scalar_bodies = crate::generate::evaluate::Plan::prepare(defs).folding_bodies(defs, types);
    for op in scalar_bodies.keys() {
        groups.entry(op).or_default();
    }
    let effectful = "!opcode.memory_effect().is_none() || opcode.is_terminator() || opcode.transfers_ownership()";
    writeln!(
        code,
        "/// Whether this opcode has a folding implementation. A particular expression may still fail to fold.\n\
         pub fn can_fold(opcode: {opcode}) -> bool {{\n\
         if {effectful} {{ return false; }}\n\
         match opcode {{"
    )
    .unwrap();
    for op in groups.keys() {
        writeln!(code, "{opcode}::{op} => true,").unwrap();
    }
    code.push_str("_ => false,\n}\n}\n");
    writeln!(code, "/// Fold a well-typed expression to constants or original operands.\n\
        /// Operand indices refer to the original argument order. A potentially\n\
        /// trapping instruction may only be erased after a successful fold.\n\
        #[allow(unused_variables, unreachable_patterns)]\n\
        pub fn fold<V: Copy + Eq>(fields: &veloc_mir::InstFields, args: &[V], results: &[{types}], mut constant: impl FnMut(V) -> Option<ScalarConst>) -> Option<smallvec::SmallVec<[crate::rewrite::Fold; 2]>> {{\n\
        let opcode = fields.opcode();\n\
        if {effectful} {{ return None; }}\n\
        match opcode {{").unwrap();
    for (op, rules) in groups {
        writeln!(code, "{opcode}::{op} => {{").unwrap();
        if let Some(rule) = rules.first() {
            let PatternKind::Operation { args, .. } = &rule.pattern[0].kind else {
                unreachable!()
            };
            writeln!(
                code,
                "debug_assert_eq!(args.len(), {}, \"fold operand count\");\nif let [ty] = results {{ let ty = *ty;",
                args.len()
            )
            .unwrap();
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
                    let mut checks = vec![crate::types::generate::accepts(
                        &rule.pattern[0].ty.domain,
                        "ty",
                    )];
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
                            PatternKind::Value(previous) if previous != slot => {
                                checks.push(format!(
                                    "crate::rewrite::same_value(args[{c}], args[{}], &mut constant)",
                                    column(previous)
                                ));
                            }
                            PatternKind::Constant(bits) => {
                                checks.push(format!(
                                    "crate::rewrite::matches_constant(constant(args[{c}]), {bits}u64, true)"
                                ));
                            }
                            _ => {}
                        }
                    }
                    if let Some(guard) = &rule.guard {
                        let guard = guard
                            .local_code(rule, types, &order)
                            .expect("locally evaluable guard");
                        // A missing fact or failed arithmetic rejects only this
                        // candidate, not later rules or operand permutations.
                        checks.push(format!("(|| Some({guard}))().unwrap_or(false)"));
                    }
                    let result = match &rule.replacement {
                        Recipe::Value(slot) => {
                            format!("crate::rewrite::Fold::Operand({})", column(*slot))
                        }
                        Recipe::Constant(_, bits) => format!(
                            "crate::rewrite::Fold::Constant(ScalarConst::from_bits(ty, {bits}u64 & u64::MAX.checked_shr(64u32.checked_sub(ty.element_bits()?)?)?)?)"
                        ),
                        _ => unreachable!(),
                    };
                    writeln!(
                        code,
                        "if {} {{ return Some(smallvec::smallvec![{result}]); }}",
                        checks.join(" && ")
                    )
                    .unwrap();
                }
            }
            code.push_str("}\n");
        }
        if let Some(body) = scalar_bodies.get(op) {
            code.push_str(body);
        } else {
            code.push_str("None\n");
        }
        code.push_str("},\n");
    }
    code.push_str("_ => None,\n} }\n");
    code
}
