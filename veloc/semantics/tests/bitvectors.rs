use veloc_semantics::{BvOp, Error, Width};

#[test]
fn modular_arithmetic_and_normalization_at_every_width() {
    for bits in 1..=128 {
        let width = Width::new(bits).unwrap();
        let mask = width.mask();
        assert_eq!(BvOp::Add.eval(bits, &[mask, 1]), Ok(0));
        assert_eq!(BvOp::Sub.eval(bits, &[0, 1]), Ok(mask));
        assert_eq!(BvOp::Mul.eval(bits, &[mask, mask]), Ok(1));
        assert_eq!(BvOp::Neg.eval(bits, &[1]), Ok(mask));
        assert_eq!(BvOp::And.eval(bits, &[u128::MAX, mask]), Ok(mask));
        assert_eq!(BvOp::Or.eval(bits, &[u128::MAX, 0]), Ok(mask));
        assert_eq!(BvOp::Xor.eval(bits, &[u128::MAX, mask]), Ok(0));
    }
    assert_eq!(BvOp::Add.eval(8, &[257, 511]), Ok(0));
    assert_eq!(BvOp::Neg.eval(128, &[1u128 << 127]), Ok(1u128 << 127));
}

#[test]
fn rejects_invalid_widths_and_operand_counts() {
    for bits in [0, 129, u16::MAX] {
        assert_eq!(
            BvOp::Add.eval(bits, &[0, 0]),
            Err(Error::InvalidWidth(bits))
        );
    }
    for op in BvOp::ALL {
        assert!(matches!(op.eval(32, &[]), Err(Error::Arity { .. })));
        assert!(matches!(op.eval(32, &[1, 2, 3]), Err(Error::Arity { .. })));
        assert_eq!(BvOp::from_name(op.name()), Some(*op));
    }
    assert_eq!(BvOp::from_name("bv.unknown"), None);
}
