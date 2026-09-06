mod common;
use common::compile_mir;
use veloc_semantics::{BvConst, BvOp};

#[test]
fn algebraic_constants_are_typed_and_names_round_trip() {
    for constant in BvConst::ALL {
        assert_eq!(BvConst::from_name(constant.name()), Some(*constant));
    }
    assert_eq!(BvConst::from_name("Two"), None);
    let error = compile_mir(&definition(BvOp::Add, "identity: Two"))
        .err()
        .unwrap();
    assert!(error.message.contains("unknown algebraic constant `Two`"));
}

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
        assert!(!generated.opcodes.contains("identity:"));
        assert!(!generated.opcodes.contains("absorbing:"));
        if facts.identity.is_some() {
            assert!(generated.evaluation.contains("Replacement::Value(args["));
        }
        if facts.absorbing.is_some() {
            assert!(
                generated
                    .evaluation
                    .contains("Replacement::Constants(alloc::vec![c])")
            );
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
