//! Cross-check concrete evaluator edge cases against the SMT encoding:
//! cargo run -q -p veloc-semantics --example check_primitives | z3 -in
use veloc_semantics::{BvOp, Error, Expr, Function, Width, equivalence_query};

fn main() -> Result<(), Error> {
    let mut expressions = Vec::new();
    let mut evaluated = Vec::new();
    for bits in [1, 8, 16, 32, 64, 128] {
        let mask = Width::new(bits)?.mask();
        let sign = 1u128 << (bits - 1);
        for &op in BvOp::ALL {
            for x in [0, 1, sign, mask] {
                for y in [0, 1, bits as u128 - 1, bits as u128, sign, mask] {
                    let args = if op.arity() == 1 { vec![x] } else { vec![x, y] };
                    let inputs = args
                        .iter()
                        .map(|&v| Expr::bv(bits, v))
                        .collect::<Result<Vec<_>, _>>()?;
                    expressions.push(Expr::apply(op, &inputs)?);
                    evaluated.push(Expr::bv(bits, op.eval(bits, &args)?)?);
                }
            }
        }
    }
    let source = Function::with_outputs([], expressions)?;
    let target = Function::with_outputs([], evaluated)?;
    print!("{}", equivalence_query(&source, &target)?);
    Ok(())
}
