use veloc_opgen::compile_mir;
use veloc_semantics::BvOp;

fn definition(op: BvOp, properties: &str) -> String {
    let (operands, args) = if op.arity() == 1 {
        ("arg: T", "arg")
    } else {
        ("lhs: T, rhs: T", "lhs, rhs")
    };
    format!(
        "format Args {{ fields: [opcode(Opcode), args(values({arity}))], opcode: dynamic(opcode) }}\n\
         op Test<T: Integer>({operands}) -> (result: T) {{
             mnemonic: \"test\", storage: Args {{ args: [{args}] }}, memory: NONE,
             semantics: {semantics}({args}), {properties}
         }}",
        arity = op.arity(),
        semantics = op.name(),
    )
}

fn rejected(op: BvOp, properties: &str, expected: &str) {
    let error = match compile_mir(&definition(op, properties)) {
        Ok(_) => panic!("contradictory algebraic declaration was accepted"),
        Err(error) => error,
    };
    assert!(error.message.contains(op.name()), "{error}");
    assert!(error.message.contains(expected), "{error}");
}

#[test]
fn rejects_traits_not_supported_by_the_semantic_primitive() {
    rejected(BvOp::Sub, "traits: [COMMUTATIVE]", "COMMUTATIVE");
    rejected(BvOp::Sub, "traits: [ASSOCIATIVE]", "ASSOCIATIVE");
    rejected(BvOp::Neg, "traits: [COMMUTATIVE]", "COMMUTATIVE");
    for op in [BvOp::Add, BvOp::Mul, BvOp::Xor] {
        rejected(
            op,
            "traits: [COMMUTATIVE, ASSOCIATIVE, IDEMPOTENT]",
            "IDEMPOTENT",
        );
    }
}

#[test]
fn rejects_incorrect_identity_and_absorbing_constants() {
    let ac = "traits: [COMMUTATIVE, ASSOCIATIVE]";
    rejected(BvOp::Add, &format!("{ac}, identity: One"), "identity `One`");
    rejected(
        BvOp::And,
        &format!("{ac}, identity: Zero"),
        "identity `Zero`",
    );
    rejected(
        BvOp::Mul,
        &format!("{ac}, absorbing: One"),
        "absorbing `One`",
    );
    rejected(
        BvOp::Or,
        &format!("{ac}, absorbing: Zero"),
        "absorbing `Zero`",
    );
    rejected(
        BvOp::Xor,
        &format!("{ac}, absorbing: Zero"),
        "absorbing `Zero`",
    );
}

#[test]
fn accepts_and_derives_the_shared_primitive_facts() {
    for op in BvOp::ALL {
        let facts = op.algebra();
        let mut traits = Vec::new();
        if facts.commutative {
            traits.push("COMMUTATIVE");
        }
        if facts.associative {
            traits.push("ASSOCIATIVE");
        }
        if facts.idempotent {
            traits.push("IDEMPOTENT");
        }
        let mut properties = format!("traits: [{}]", traits.join(", "));
        if let Some(identity) = facts.identity {
            properties.push_str(&format!(", identity: {}", identity.name()));
        }
        if let Some(absorbing) = facts.absorbing {
            properties.push_str(&format!(", absorbing: {}", absorbing.name()));
        }
        compile_mir(&definition(*op, &properties)).unwrap();
        let generated = compile_mir(&definition(*op, "traits: []")).unwrap();
        for name in traits {
            assert!(generated.opcodes.contains(&format!("OpTraits::{name}")));
        }
        if let Some(identity) = facts.identity {
            assert!(generated.opcodes.contains(&format!(
                "identity: Some(crate::opspec::AlgebraicConstant::{identity:?})"
            )));
        }
        if let Some(absorbing) = facts.absorbing {
            assert!(generated.opcodes.contains(&format!(
                "absorbing: Some(crate::opspec::AlgebraicConstant::{absorbing:?})"
            )));
        }
    }
}

#[test]
fn operations_without_semantics_keep_explicit_unverified_facts() {
    let source = definition(
        BvOp::Add,
        "traits: [COMMUTATIVE, ASSOCIATIVE], identity: One",
    )
    .replace("semantics: bv.add(lhs, rhs),", "");
    // Without a semantic definition there is nothing to cross-check. This is
    // still an author-supplied contract, not a claim of automatic verification.
    compile_mir(&source).unwrap();
}

#[test]
fn composed_semantics_do_not_inherit_the_root_primitive_facts() {
    let source =
        definition(BvOp::Neg, "traits: []").replace("bv.neg(arg)", "bv.sub(bv.zero(), arg)");
    let generated = compile_mir(&source).unwrap();
    assert!(
        generated
            .opcodes
            .contains("crate::semantics::Program { inputs: 1")
    );
    assert!(
        generated
            .opcodes
            .contains("Step::Const(crate::semantics::BvConst::Zero)")
    );
    assert!(
        generated
            .opcodes
            .contains("op: crate::semantics::BvOp::Sub")
    );
    assert!(!generated.opcodes.contains("OpTraits::COMMUTATIVE"));
    assert!(!generated.opcodes.contains("OpTraits::ASSOCIATIVE"));

    let nested = definition(BvOp::Add, "traits: []")
        .replace("bv.add(lhs, rhs)", "bv.add(lhs, bv.add(rhs, bv.one()))");
    let generated = compile_mir(&nested).unwrap();
    assert_eq!(
        generated
            .opcodes
            .matches("op: crate::semantics::BvOp::Add")
            .count(),
        2
    );
    assert!(generated.opcodes.contains("identity: None"));
    assert!(!generated.opcodes.contains("OpTraits::ASSOCIATIVE"));
}
