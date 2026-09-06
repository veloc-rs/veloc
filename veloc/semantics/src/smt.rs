use std::fmt::Write;

use crate::expr::Node;
use crate::{BvOp, Error, Expr, Function, Sort, Trap};

/// Emit a QF_BV query asking for an input on which two functions disagree.
///
/// `unsat` establishes equality of values AND trap outcomes in this model;
/// `sat` reports a counterexample exists; `unknown` is not a proof. The encoder
/// and primitive semantics are part of the trusted boundary. No solver is run.
/// Both input signatures and output sorts must match exactly.
pub fn equivalence_query(source: &Function, target: &Function) -> Result<String, Error> {
    if source.inputs() != target.inputs() {
        return Err(Error::SignatureMismatch);
    }
    if source.outputs().len() != target.outputs().len() {
        return Err(Error::ResultArity {
            expected: source.outputs().len(),
            actual: target.outputs().len(),
        });
    }
    for (source, target) in source.outputs().iter().zip(target.outputs()) {
        if source.sort() != target.sort() {
            return Err(Error::SortMismatch {
                expected: source.sort(),
                actual: target.sort(),
            });
        }
    }
    let mut text = String::from(
        "; Counterexample query: unsat means equal in this model.\n(set-logic QF_BV)\n",
    );
    for (index, sort) in source.inputs().iter().enumerate() {
        writeln!(text, "(declare-fun x{index} () {})", smt_sort(*sort)).unwrap();
    }
    emit_function(&mut text, source, "s");
    emit_function(&mut text, target, "t");
    let differences = source
        .outputs()
        .iter()
        .zip(target.outputs())
        .map(|(s, t)| {
            format!(
                "(not (= s{} t{}))",
                source.indices[&s.key()],
                target.indices[&t.key()]
            )
        })
        .collect::<Vec<_>>();
    if !source.traps().is_empty() || !target.traps().is_empty() {
        emit_trap(&mut text, source, "s");
        emit_trap(&mut text, target, "t");
        let values_differ = if differences.is_empty() {
            "false".into()
        } else {
            format!("(or {})", differences.join(" "))
        };
        writeln!(
            text,
            "(assert (or (not (= s_trap t_trap)) (and (= s_trap (_ bv0 8)) {values_differ})))"
        )
        .unwrap();
    } else {
        match differences.as_slice() {
            [] => text.push_str("(assert false)\n"),
            [difference] => writeln!(text, "(assert {difference})").unwrap(),
            _ => writeln!(text, "(assert (or {}))", differences.join(" ")).unwrap(),
        }
    }
    text.push_str("(check-sat)\n");
    Ok(text)
}

fn smt_sort(sort: Sort) -> String {
    match sort {
        Sort::Bool => "Bool".into(),
        Sort::Bv(width) => format!("(_ BitVec {})", width.bits()),
    }
}

fn emit_trap(text: &mut String, function: &Function, prefix: &str) {
    let mut result = "(_ bv0 8)".to_string();
    for (guard, trap) in function.traps().iter().rev() {
        let code = match trap {
            Trap::DivisionByZero => 1,
            Trap::IntegerOverflow => 2,
        };
        result = format!(
            "(ite {prefix}{} (_ bv{code} 8) {result})",
            function.indices[&guard.key()]
        );
    }
    writeln!(text, "(define-fun {prefix}_trap () (_ BitVec 8) {result})").unwrap();
}

// Portable QF_BV encodings; no solver-specific count operators are required.
fn bit_count(op: BvOp, width: u16, arg: &str) -> String {
    let number = |n| format!("(_ bv{n} {width})");
    match op {
        BvOp::Clz | BvOp::Ctz => {
            let mut result = number(width);
            for i in 0..width {
                let bit = if op == BvOp::Clz { i } else { width - 1 - i };
                let count = if op == BvOp::Clz {
                    width - 1 - bit
                } else {
                    bit
                };
                result = format!(
                    "(ite (= ((_ extract {bit} {bit}) {arg}) #b1) {} {result})",
                    number(count)
                );
            }
            result
        }
        BvOp::Popcnt => (0..width).fold(number(0), |sum, bit| {
            format!(
                "(bvadd {sum} ((_ zero_extend {}) ((_ extract {bit} {bit}) {arg})))",
                width - 1
            )
        }),
        _ => unreachable!("only bit counts lack a native SMT operator"),
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
                let names = args.iter().map(name).collect::<Vec<_>>();
                if expr.sort() == Sort::Bool {
                    let operation = match op {
                        crate::BvOp::And => "and",
                        crate::BvOp::Or => "or",
                        crate::BvOp::Xor => "xor",
                        _ => unreachable!("typed bool operation"),
                    };
                    format!("({operation} {})", names.join(" "))
                } else if let Some(operation) = op.smt_name() {
                    format!("({operation} {})", names.join(" "))
                } else {
                    bit_count(*op, expr.sort().width().unwrap().bits(), &names[0])
                }
            }
            Node::Concat(high, low) => format!("(concat {} {})", name(high), name(low)),
            Node::Extract { arg, high, low } => format!("((_ extract {high} {low}) {})", name(arg)),
            Node::ZeroExtend(arg) => {
                let extra =
                    expr.sort().width().unwrap().bits() - arg.sort().width().unwrap().bits();
                format!("((_ zero_extend {extra}) {})", name(arg))
            }
            Node::SignExtend(arg) => {
                let extra =
                    expr.sort().width().unwrap().bits() - arg.sort().width().unwrap().bits();
                format!("((_ sign_extend {extra}) {})", name(arg))
            }
            Node::Compare {
                predicate,
                lhs,
                rhs,
            } => {
                let order = if predicate.signed() { "s" } else { "u" };
                let mut terms = Vec::new();
                for (bit, op) in [
                    (1, format!("bv{order}lt")),
                    (2, "=".into()),
                    (4, format!("bv{order}gt")),
                ] {
                    if predicate.outcomes() & bit != 0 {
                        terms.push(format!("({op} {} {})", name(lhs), name(rhs)));
                    }
                }
                match terms.as_slice() {
                    [] => "false".into(),
                    [term] => term.clone(),
                    _ => format!("(or {})", terms.join(" ")),
                }
            }
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
