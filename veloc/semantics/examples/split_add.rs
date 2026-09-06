//! Print a counterexample query for i64 addition lowered to pairs of i32s.
//!
//! cargo run -q -p veloc-semantics --example split_add | z3 -in
//!
//! Expected: unsat. Add --broken to omit the carry and obtain sat instead.

use veloc_semantics::{BvOp, Error, Expr, Function, Sort, equivalence_query};

fn main() -> Result<(), Error> {
    let sort = Sort::bv(64)?;
    let x = Expr::input(0, sort);
    let y = Expr::input(1, sort);
    let source = Function::new(
        [sort, sort],
        Expr::apply(BvOp::Add, &[x.clone(), y.clone()])?,
    )?;

    let xl = x.extract(31, 0)?;
    let yl = y.extract(31, 0)?;
    let xh = x.extract(63, 32)?;
    let yh = y.extract(63, 32)?;
    let lo = Expr::apply(BvOp::Add, &[xl.clone(), yl])?;
    let mut hi = Expr::apply(BvOp::Add, &[xh, yh])?;
    if !std::env::args().any(|arg| arg == "--broken") {
        let carry = lo.ult(&xl)?.bool_to_bv(1)?.zero_extend(32)?;
        hi = Expr::apply(BvOp::Add, &[hi, carry])?;
    }
    let target = Function::new([sort, sort], Expr::concat(&hi, &lo)?)?;
    print!("{}", equivalence_query(&source, &target)?);
    Ok(())
}
