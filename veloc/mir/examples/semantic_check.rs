//! Export checks of actual defs-driven semantics. A solver is optional:
//! cargo run -q -p veloc-mir --example semantic_check -- overflow | z3 -in
//! cargo run -q -p veloc-mir --example semantic_check -- overflow --broken | z3 -in
use veloc_mir::semantics::{
    BvOp, Error, Expr, Function, IntPredicate, Sort, Trap, equivalence_query,
};
use veloc_mir::{IntCC, Opcode};

fn main() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mode = args.first().map(String::as_str).unwrap_or("overflow");
    // Keep the widened division query small enough for an interactive check.
    let bits = if mode == "division" { 8 } else { 32 };
    let sort = Sort::bv(bits)?;
    let x = Expr::input(0, sort);
    let y = Expr::input(1, sort);
    let (source, target) =
        match mode {
            "division" => {
                let source = Opcode::IDivS.spec().semantics.unwrap().instantiate(
                    &[sort, sort],
                    &[sort],
                    &[],
                )?;
                // Widen before division: overflow is exactly a quotient that cannot
                // be reconstructed by truncation followed by sign extension.
                let quotient = Expr::apply(
                    BvOp::SDiv,
                    &[x.sign_extend(bits * 2)?, y.sign_extend(bits * 2)?],
                )?;
                let result = quotient.extract(bits - 1, 0)?;
                let zero = y.compare(&Expr::bv(bits, 0)?, IntPredicate::new(false, 2))?;
                let overflow = quotient
                    .compare(&result.sign_extend(bits * 2)?, IntPredicate::new(false, 5))?;
                let traps = if args.iter().any(|a| a == "--broken") {
                    vec![(zero, Trap::DivisionByZero)]
                } else {
                    vec![
                        (zero, Trap::DivisionByZero),
                        (overflow, Trap::IntegerOverflow),
                    ]
                };
                (source, Function::with_traps([sort, sort], [result], traps)?)
            }
            "shift" => {
                let source = Opcode::IShrS.spec().semantics.unwrap().instantiate(
                    &[sort, sort],
                    &[sort],
                    &[],
                )?;
                let count = Expr::apply(BvOp::And, &[y, Expr::bv(32, 31)?])?;
                (
                    source,
                    Function::new([sort, sort], Expr::apply(BvOp::AShr, &[x, count])?)?,
                )
            }
            "overflow" => {
                let source = Opcode::IAddWithOverflow
                    .spec()
                    .semantics
                    .unwrap()
                    .instantiate(&[sort, sort], &[sort, Sort::Bool], &[])?;
                // Independent definition: add in twice the width, then check whether
                // truncation followed by signed extension reproduces the exact sum.
                let wide = Expr::apply(
                    BvOp::Add,
                    &[x.sign_extend(bits * 2)?, y.sign_extend(bits * 2)?],
                )?;
                let sum = wide.extract(bits - 1, 0)?;
                let overflow = if args.iter().any(|a| a == "--broken") {
                    Expr::bool(false)
                } else {
                    wide.compare(&sum.sign_extend(bits * 2)?, IntPredicate::new(false, 5))?
                };
                (
                    source,
                    Function::with_outputs([sort, sort], [sum, overflow])?,
                )
            }
            "comparison" => {
                let source = Opcode::Icmp.spec().semantics.unwrap().instantiate(
                    &[sort, sort],
                    &[Sort::Bool],
                    &[IntCC::LtS.predicate()],
                )?;
                let sign = Expr::bv(bits, 1 << (bits - 1))?;
                let lhs = Expr::apply(BvOp::Xor, &[x, sign.clone()])?;
                let rhs = Expr::apply(BvOp::Xor, &[y, sign])?;
                (source, Function::new([sort, sort], lhs.ult(&rhs)?)?)
            }
            "zext" | "sext" | "trunc" => {
                let (op, width) = match mode {
                    "zext" => (Opcode::ExtendU, 64),
                    "sext" => (Opcode::ExtendS, 64),
                    _ => (Opcode::Wrap, 16),
                };
                let source =
                    op.spec()
                        .semantics
                        .unwrap()
                        .instantiate(&[sort], &[Sort::bv(width)?], &[])?;
                let result = match mode {
                    "zext" => Expr::concat(&Expr::bv(32, 0)?, &x)?,
                    "sext" => {
                        let negative = x.compare(&Expr::bv(32, 0)?, IntPredicate::new(true, 1))?;
                        let high = Expr::select(
                            &negative,
                            &Expr::bv(32, u32::MAX as u128)?,
                            &Expr::bv(32, 0)?,
                        )?;
                        Expr::concat(&high, &x)?
                    }
                    _ => x.extract(15, 0)?,
                };
                (source, Function::new([sort], result)?)
            }
            _ => panic!("expected overflow, division, shift, comparison, zext, sext or trunc"),
        };
    print!("{}", equivalence_query(&source, &target)?);
    Ok(())
}
