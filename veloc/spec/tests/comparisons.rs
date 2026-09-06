mod common;
use common::compile_mir;

fn rejected(domain: &str, predicates: &str, message: &str) {
    let source = format!("comparison TestCC {{ domain: {domain}, predicates: [{predicates}] }}");
    let error = compile_mir(&source)
        .err()
        .expect("invalid comparison accepted");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn transforms_follow_outcomes_not_variant_names() {
    let output = compile_mir(
        r#"
        comparison TestCC {
            domain: integer,
            predicates: [
                Before(signed, [less]), After(signed, [greater]),
                NotBefore(signed, [equal, greater]), NotAfter(signed, [less, equal]),
            ],
        }
    "#,
    )
    .unwrap();
    let code = output.opcodes.split("impl TestCC {").nth(1).unwrap();
    let swap = code
        .split("pub const fn swap")
        .nth(1)
        .unwrap()
        .split("pub const fn complement")
        .next()
        .unwrap();
    assert!(swap.contains("Self::Before => Self::After,"));
    assert!(swap.contains("Self::NotBefore => Self::NotAfter,"));
    let complement = code.split("pub const fn complement").nth(1).unwrap();
    assert!(complement.contains("Self::Before => Self::NotBefore,"));
    assert!(complement.contains("Self::After => Self::NotAfter,"));
    assert!(code.contains("\"notbefore\" => Some(Self::NotBefore),"));
}

#[test]
fn unordered_predicates_enable_exact_float_complements() {
    let output = compile_mir(
        r#"
        comparison TestCC {
            domain: float,
            predicates: [
                Before([less]), After([greater]),
                NotBefore([equal, greater, unordered]),
                NotAfter([less, equal, unordered]),
            ],
        }
    "#,
    )
    .unwrap();
    let code = output.opcodes.split("impl TestCC {").nth(1).unwrap();
    let complement = code.split("pub const fn complement(self)").nth(1).unwrap();
    assert!(complement.contains("Self::Before => Some(Self::NotBefore),"));
    assert!(complement.contains("Self::NotAfter => Some(Self::After),"));
}

#[test]
fn invalid_outcomes_and_signedness_are_rejected() {
    rejected("integer", "Lt([less])", "requires signedness");
    rejected(
        "integer",
        "Eq(signed, [equal])",
        "equality predicates must omit",
    );
    rejected(
        "integer",
        "Lt(other, [less])",
        "expected signed or unsigned",
    );
    rejected("integer", "Eq([unordered])", "invalid comparison outcome");
    rejected(
        "float",
        "Lt(signed, [less])",
        "optionally preceded by integer",
    );
    rejected("float", "Eq([unknown])", "invalid comparison outcome");
    rejected(
        "float",
        "Eq([equal, equal])",
        "duplicate comparison outcome",
    );
    rejected("other", "", "expected integer or float");
    rejected("float", "", "at least one predicate");
    rejected("float", "Self([equal])", "invalid generated identifier");
}

#[test]
fn ambiguous_predicates_are_rejected() {
    rejected(
        "float",
        "Eq([equal]), EQ([less])",
        "duplicate comparison mnemonic",
    );
    rejected(
        "float",
        "Eq([equal]), Same([equal])",
        "duplicate comparison semantics",
    );
}

#[test]
fn required_transforms_must_be_represented_in_the_same_domain() {
    rejected(
        "integer",
        "Lt(signed, [less]), Gt(unsigned, [greater])",
        "no swap",
    );
    rejected("integer", "Eq([equal])", "no complement");
    rejected("float", "Lt([less])", "no swap");
    rejected("float", "Eq([equal])", "no ordered complement");
}

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
