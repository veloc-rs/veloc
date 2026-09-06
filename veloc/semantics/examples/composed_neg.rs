//! Compare a composed semantic program, sub(Zero, x), with primitive negation.
//!
//! cargo run -q -p veloc-semantics --example composed_neg | z3 -in
//!
//! Expected: unsat, for the selected concrete width (64 bits).

use veloc_semantics::{BvConst, BvOp, Error, Program, Step, Width, equivalence_query};

const NEG: Program = Program {
    inputs: 1,
    steps: &[
        Step::Input(0),
        Step::Apply {
            op: BvOp::Neg,
            args: &[0],
        },
    ],
    output: 1,
};

const COMPOSED_NEG: Program = Program {
    inputs: 1,
    steps: &[
        Step::Const(BvConst::Zero),
        Step::Input(0),
        Step::Apply {
            op: BvOp::Sub,
            args: &[0, 1],
        },
    ],
    output: 2,
};

fn main() -> Result<(), Error> {
    let width = Width::new(64)?;
    print!(
        "{}",
        equivalence_query(&NEG.instantiate(width)?, &COMPOSED_NEG.instantiate(width)?)?
    );
    Ok(())
}
