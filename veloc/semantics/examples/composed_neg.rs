//! Compare a composed semantic program, sub(Zero, x), with primitive negation.
//!
//! cargo run -q -p veloc-semantics --example composed_neg | z3 -in
//!
//! Expected: unsat, for the selected concrete width (64 bits).

use veloc_semantics::{BvConst, BvOp, Error, Program, Step, Width, equivalence_query};

const NEG: Program = Program {
    inputs: 1,
    properties: 0,
    traps: &[],
    steps: &[
        Step::Input(0),
        Step::Apply {
            op: BvOp::Neg,
            args: &[0],
        },
    ],
    outputs: &[1],
};

const COMPOSED_NEG: Program = Program {
    inputs: 1,
    properties: 0,
    traps: &[],
    steps: &[
        Step::Const {
            value: BvConst::Zero,
            ty: veloc_semantics::TypeRef::Result(0),
        },
        Step::Input(0),
        Step::Apply {
            op: BvOp::Sub,
            args: &[0, 1],
        },
    ],
    outputs: &[2],
};

fn main() -> Result<(), Error> {
    let width = Width::new(64)?;
    let sorts = [veloc_semantics::Sort::Bv(width)];
    print!(
        "{}",
        equivalence_query(
            &NEG.instantiate(&sorts, &sorts, &[])?,
            &COMPOSED_NEG.instantiate(&sorts, &sorts, &[])?
        )?
    );
    Ok(())
}
