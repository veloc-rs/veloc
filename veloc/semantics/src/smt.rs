use std::fmt::Write;

use crate::expr::Node;
use crate::{Error, Expr, Function, Sort};

/// Emit a QF_BV query asking for an input on which two functions disagree.
///
/// `unsat` establishes equality for this fixed-width, pure expression model;
/// `sat` reports a counterexample exists; `unknown` is not a proof. The encoder
/// and primitive semantics are part of the trusted boundary. No solver is run.
/// Both input signatures and output sorts must match exactly.
pub fn equivalence_query(source: &Function, target: &Function) -> Result<String, Error> {
    if source.inputs() != target.inputs() {
        return Err(Error::SignatureMismatch);
    }
    if source.output_sort() != target.output_sort() {
        return Err(Error::SortMismatch {
            expected: source.output_sort(),
            actual: target.output_sort(),
        });
    }
    let mut text = String::from(
        "; Counterexample query: unsat means equal in this model.\n(set-logic QF_BV)\n",
    );
    for (index, sort) in source.inputs().iter().enumerate() {
        writeln!(text, "(declare-fun x{index} () {})", smt_sort(*sort)).unwrap();
    }
    emit_function(&mut text, source, "s");
    emit_function(&mut text, target, "t");
    writeln!(
        text,
        "(assert (not (= s{} t{})))",
        source.nodes.len() - 1,
        target.nodes.len() - 1
    )
    .unwrap();
    text.push_str("(check-sat)\n");
    Ok(text)
}

fn smt_sort(sort: Sort) -> String {
    match sort {
        Sort::Bool => "Bool".into(),
        Sort::Bv(width) => format!("(_ BitVec {})", width.bits()),
    }
}

fn emit_function(text: &mut String, function: &Function, prefix: &str) {
    let name = |expr: &Expr| format!("{prefix}{}", function.indices[&expr.key()]);
    for (index, expr) in function.nodes.iter().enumerate() {
        let body = match expr.node() {
            Node::Input(index) => format!("x{index}"),
            Node::Bv(value) => format!("(_ bv{value} {})", expr.sort().width().unwrap().bits()),
            Node::Bool(value) => value.to_string(),
            Node::Apply { op, args } => {
                let args = args.iter().map(name).collect::<Vec<_>>().join(" ");
                format!("({} {args})", op.smt_name())
            }
            Node::Concat(high, low) => format!("(concat {} {})", name(high), name(low)),
            Node::Extract { arg, high, low } => format!("((_ extract {high} {low}) {})", name(arg)),
            Node::ZeroExtend(arg) => {
                let extra =
                    expr.sort().width().unwrap().bits() - arg.sort().width().unwrap().bits();
                format!("((_ zero_extend {extra}) {})", name(arg))
            }
            Node::Ult(a, b) => format!("(bvult {} {})", name(a), name(b)),
            Node::Select { cond, yes, no } => {
                format!("(ite {} {} {})", name(cond), name(yes), name(no))
            }
        };
        // Shared subexpressions get one definition; output size is linear in
        // graph size rather than exponential in the expanded expression tree.
        writeln!(
            text,
            "(define-fun {prefix}{index} () {} {body})",
            smt_sort(expr.sort())
        )
        .unwrap();
    }
}
