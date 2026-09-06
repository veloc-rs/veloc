mod common;
use common::compile_mir;

#[test]
fn every_float_outcome_set_has_the_expected_complement() {
    // Exercise all 16 sets, including empty/always, ordered/unordered, and
    // predicates equivalent only under the non-NaN assumption.
    let outcomes = ["less", "equal", "greater", "unordered"];
    let predicates = (0..16)
        .map(|bits| {
            let accepted = outcomes
                .iter()
                .enumerate()
                .filter(|(bit, _)| bits & (1 << bit) != 0)
                .map(|(_, name)| *name)
                .collect::<Vec<_>>()
                .join(", ");
            format!("P{bits}([{accepted}])")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let output = compile_mir(&format!(
        "comparison TestCC {{ domain: float, predicates: [{predicates}] }}"
    ))
    .unwrap();
    let code = output.opcodes.split("impl TestCC {").nth(1).unwrap();
    let complement = code.split("pub const fn complement(self)").nth(1).unwrap();
    let ordered = code
        .split("pub const fn complement_ordered")
        .nth(1)
        .unwrap();
    let swap = code.split("pub const fn swap").nth(1).unwrap();
    for bits in 0..16 {
        assert!(complement.contains(&format!("Self::P{bits} => Some(Self::P{}),", bits ^ 15)));
        assert!(ordered.contains(&format!("Self::P{bits} => Self::P{},", (bits & 7) ^ 7)));
        let swapped = (bits & 10) | ((bits & 1) << 2) | ((bits & 4) >> 2);
        assert!(swap.contains(&format!("Self::P{bits} => Self::P{swapped},")));
    }
}
