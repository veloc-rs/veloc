//! Rust owns foreign constant evaluation. Candidate structure comes from defs;
//! applicability and assertions are evaluated by rustc, not by a query registry.
use std::collections::BTreeMap;
use std::fmt::Write;

use crate::model::{Definitions, Op, ParamKind, expr::Emitter};

pub(crate) fn generate(defs: &Definitions) -> String {
    let mut out = String::from(
        "#[allow(non_upper_case_globals, unused_variables, unused_parens, unused_imports)]\npub mod semantic_cases {\nuse crate::Type;\n",
    );
    for op in &defs.ops {
        for (check, message) in &op.meta.checks {
            writeln!(
                out,
                "const _: () = assert!({}, {:?});",
                check.const_rust("crate::inst::"),
                format!("{}: {message}", op.name)
            )
            .unwrap();
        }
        let Some(sem) = &op.semantics else {
            check_signature(&mut out, defs, op);
            continue;
        };
        let constraints = op
            .constraints
            .iter()
            .filter(|c| c.type_only && !c.condition.is_bool(true))
            .collect::<Vec<_>>();
        let count = sem.instances.len();
        if constraints.is_empty() {
            let values = sem
                .instances
                .iter()
                .map(|i| i.scalar.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "pub const {}: [bool; {count}] = [{values}];", op.name).unwrap();
            continue;
        }
        writeln!(out, "pub const {}: [bool; {count}] = {{", op.name).unwrap();
        out.push_str("const fn applies(operands: &[Type], results: &[Type]) -> bool {\n");
        let emitter = type_emitter(op);
        for constraint in constraints {
            writeln!(
                out,
                "if !({}) {{ return false; }}",
                emitter.term(&constraint.condition)
            )
            .unwrap();
        }
        out.push_str("true\n}\n");
        writeln!(
            out,
            "let mut enabled = [false; {count}]; let mut any = false;"
        )
        .unwrap();
        for (index, instance) in sem.instances.iter().enumerate() {
            let loops = enter_case(
                &mut out,
                defs,
                &instance.kinds,
                &instance.shapes,
                &instance.same,
            );
            let inputs = sem.inputs as usize;
            let values = |range: std::ops::Range<usize>| {
                range
                    .map(|n| format!("ty{n}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            writeln!(
                out,
                "if applies(&[{}], &[{}]) {{",
                values(0..inputs),
                values(inputs..instance.kinds.len())
            )
            .unwrap();
            for slot in 1..instance.kinds.len() {
                writeln!(
                    out,
                    "assert!(shape{slot} == shape0, {:?});",
                    format!("{}: semantic recipes require a shared lane shape", op.name)
                )
                .unwrap();
            }
            if let Some(error) = &instance.error {
                writeln!(out, "panic!({:?});", format!("{}: {error}", op.name)).unwrap();
            } else {
                out.push_str("any = true;\n");
                let scalar = if instance.kinds.is_empty() {
                    "true"
                } else {
                    "shape0 == 0"
                };
                writeln!(out, "if {scalar} {{ enabled[{index}] = true; }}").unwrap();
            }
            out.push_str("}\n");
            for _ in 0..loops {
                out.push_str("}\n");
            }
            out.push_str("}\n");
        }
        writeln!(
            out,
            "assert!(any, {:?}); enabled\n}};",
            format!("{}: no admissible semantic signature", op.name)
        )
        .unwrap();
    }
    out.push_str("}\n");
    out
}

pub(crate) fn type_emitter(op: &Op) -> Emitter<'_> {
    let mut emitter = Emitter::query(BTreeMap::new());
    emitter.constant = true;
    emitter.const_failure = "return false";
    emitter.results = "results";
    emitter.result_values = false;
    emitter.operand_types = op
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Value)
        .enumerate()
        .map(|(i, p)| (p.name.clone(), format!("operands[{i}]")))
        .collect();
    emitter
}

fn check_signature(out: &mut String, defs: &Definitions, op: &Op) {
    let constraints = op
        .constraints
        .iter()
        .filter(|c| {
            c.type_only && !c.condition.is_bool(true) && c.condition.const_type_query(&op.params)
        })
        .collect::<Vec<_>>();
    if constraints.is_empty() {
        return;
    }
    // These are optional early diagnostics. Structural/dynamic signatures remain
    // checked by the generated validator, without sampling their type domains.
    let Ok(Some(cases)) = crate::types::cases::enumerate(&defs.types, &op.signature) else {
        return;
    };
    writeln!(
        out,
        "const _: () = {{ const fn applies(operands: &[Type], results: &[Type]) -> bool {{"
    )
    .unwrap();
    let emitter = type_emitter(op);
    for constraint in constraints {
        writeln!(
            out,
            "if !({}) {{ return false; }}",
            emitter.term(&constraint.condition)
        )
        .unwrap();
    }
    out.push_str("true }\nlet possible = 'search: {\n");
    for case in cases {
        let loops = enter_case(out, defs, &case.kinds, &case.shapes, &case.same);
        let inputs = op.signature.operands.patterns().unwrap().len();
        let values = |range: std::ops::Range<usize>| {
            range
                .map(|n| format!("ty{n}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            out,
            "if applies(&[{}], &[{}]) {{ break 'search true; }}",
            values(0..inputs),
            values(inputs..case.kinds.len())
        )
        .unwrap();
        for _ in 0..loops {
            out.push_str("}\n");
        }
        out.push_str("}\n");
    }
    writeln!(
        out,
        "false }}; assert!(possible, {:?}); }};",
        format!("{}: no admissible type signature", op.name)
    )
    .unwrap();
}

/// Emit the shared shape loops for one element-kind signature.
fn enter_case(
    out: &mut String,
    defs: &Definitions,
    kinds: &[crate::types::Primitive],
    shapes: &[u32],
    same: &[Option<usize>],
) -> usize {
    let mut loops = 0;
    out.push_str("{\n");
    for (slot, mask) in shapes.iter().enumerate() {
        if let Some(other) = same[slot] {
            writeln!(out, "let shape{slot} = shape{other};").unwrap();
        } else {
            writeln!(out, "let mut mask{slot} = {mask}u32; while mask{slot} != 0 {{ let shape{slot} = mask{slot}.trailing_zeros(); mask{slot} &= mask{slot} - 1;").unwrap();
            loops += 1;
        }
        let scalar = defs
            .types
            .scalars
            .iter()
            .find(|s| s.ty == kinds[slot])
            .unwrap()
            .exact();
        if *mask == 1 {
            writeln!(out, "let ty{slot} = Type::{scalar};").unwrap();
        } else {
            writeln!(out, "let ty{slot} = if shape{slot} == 0 {{ Type::{scalar} }} else {{ veloc_types::ScalarType::{scalar}.vector(1u16 << (shape{slot} % 16), shape{slot} >= 16).expect(\"declared vector shape\").as_type() }};").unwrap();
        }
    }

    loops
}
