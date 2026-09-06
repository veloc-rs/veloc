//! Build-time reference documentation and static type diagnostics.
use std::fmt::Write;

use crate::model::{Definitions, Pattern, TypeList};
use crate::type_gen::Classes;

pub(crate) fn pattern(p: &Pattern, classes: &Classes) -> String {
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

fn type_list(list: &TypeList, classes: &Classes) -> String {
    let (TypeList::Fixed(patterns) | TypeList::Variadic(patterns)) = list else {
        return "signature".into();
    };
    let mut types = patterns
        .iter()
        .map(|p| pattern(p, classes))
        .collect::<Vec<_>>();
    if matches!(list, TypeList::Variadic(_)) {
        types.push("...".into());
    }
    format!("({})", types.join(", "))
}

pub(crate) fn generate(defs: &Definitions, classes: &Classes) -> String {
    let mut output = String::from(
        "# MIR operation reference\n\nGenerated from checked operation definitions. `T0`, `T1`, etc. are local to each signature.\n\n| Mnemonic | Format | Operands | Results | Relations | Memory | Properties | Constraints |\n|---|---|---|---|---|---|---|---|\n",
    );
    let cell = |text: String| text.replace('|', "\\|").replace('\n', " ");
    for op in &defs.ops {
        let relations = op
            .signature
            .relations
            .iter()
            .map(|r| {
                let slot = |s: crate::model::Slot| {
                    format!(
                        "{}[{}]",
                        if s.result { "results" } else { "operands" },
                        s.index
                    )
                };
                format!("{}({}, {})", r.kind, slot(r.lhs), slot(r.rhs))
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | `{}` |",
            op.mnemonic,
            op.format,
            cell(type_list(&op.signature.operands, classes)),
            cell(type_list(&op.signature.results, classes)),
            cell(relations),
            op.memory,
            op.traits.join(", "),
            cell(
                op.constraints
                    .iter()
                    .map(|c| c.text.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
        .unwrap();
    }
    output
}
