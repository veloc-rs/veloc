use veloc_semantics::{Algebra, BvConst, BvOp, Error, Width};

#[test]
fn primitive_facts_hold_exhaustively_at_small_widths() {
    for op in BvOp::ALL {
        let facts = op.algebra();
        if op.arity() != 2 {
            assert_eq!(facts, Algebra::NONE);
            continue;
        }
        for width in 1..=4 {
            let max = Width::new(width).unwrap().mask();
            let eval = |x, y| op.eval(width, &[x, y]).unwrap();
            for x in 0..=max {
                if facts.idempotent {
                    assert_eq!(eval(x, x), x);
                }
                if let Some(identity) = facts.identity {
                    let identity = identity.eval(width).unwrap();
                    assert_eq!(eval(x, identity), x);
                    assert_eq!(eval(identity, x), x);
                }
                if let Some(absorbing) = facts.absorbing {
                    let absorbing = absorbing.eval(width).unwrap();
                    assert_eq!(eval(x, absorbing), absorbing);
                    assert_eq!(eval(absorbing, x), absorbing);
                }
                for y in 0..=max {
                    if facts.commutative {
                        assert_eq!(eval(x, y), eval(y, x));
                    }
                    if facts.associative {
                        for z in 0..=max {
                            assert_eq!(eval(eval(x, y), z), eval(x, eval(y, z)));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn constants_and_two_sided_elements_cover_all_supported_widths() {
    for width in 1..=128 {
        let mask = Width::new(width).unwrap().mask();
        assert_eq!(BvConst::AllOnes.eval(width), Ok(mask));
        assert_eq!(BvConst::Zero.eval(width), Ok(0));
        assert_eq!(BvConst::One.eval(width), Ok(1));
        for op in BvOp::ALL {
            let facts = op.algebra();
            for x in [0, 1, mask / 2, mask] {
                if let Some(identity) = facts.identity {
                    let identity = identity.eval(width).unwrap();
                    assert_eq!(op.eval(width, &[x, identity]), Ok(x));
                    assert_eq!(op.eval(width, &[identity, x]), Ok(x));
                }
                if let Some(absorbing) = facts.absorbing {
                    let absorbing = absorbing.eval(width).unwrap();
                    assert_eq!(op.eval(width, &[x, absorbing]), Ok(absorbing));
                    assert_eq!(op.eval(width, &[absorbing, x]), Ok(absorbing));
                }
            }
        }
    }
    assert_eq!(BvConst::Zero.eval(0), Err(Error::InvalidWidth(0)));
    assert_eq!(BvConst::One.eval(129), Err(Error::InvalidWidth(129)));
}

#[test]
fn width_specific_or_one_sided_facts_are_not_generalized() {
    assert_eq!(BvOp::Sub.algebra(), Algebra::NONE);
    assert_eq!(BvOp::Neg.algebra(), Algebra::NONE);
    assert!(!BvOp::Add.algebra().idempotent);
    assert!(!BvOp::Mul.algebra().idempotent);
    assert!(!BvOp::Xor.algebra().idempotent);
    assert_eq!(BvOp::Add.algebra().absorbing, None);
    assert_eq!(BvOp::Xor.algebra().absorbing, None);
    assert_eq!(BvOp::And.algebra().identity, Some(BvConst::AllOnes));
    assert_eq!(BvOp::Or.algebra().absorbing, Some(BvConst::AllOnes));
}
