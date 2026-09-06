use veloc_semantics::{BvOp, Error, Width};

#[test]
fn raw_primitives_are_total_and_do_not_mask_shift_counts() {
    for bits in 1..=128 {
        let mask = Width::new(bits).unwrap().mask();
        let min = 1u128 << (bits - 1);
        for x in [0, 1, min, mask] {
            assert_eq!(BvOp::UDiv.eval(bits, &[x, 0]), Ok(mask));
            assert_eq!(BvOp::URem.eval(bits, &[x, 0]), Ok(x));
            assert_eq!(
                BvOp::SDiv.eval(bits, &[x, 0]),
                Ok(if x & min != 0 { 1 } else { mask })
            );
            assert_eq!(BvOp::SRem.eval(bits, &[x, 0]), Ok(x));
        }
        assert_eq!(BvOp::SDiv.eval(bits, &[min, mask]), Ok(min));
        assert_eq!(BvOp::SRem.eval(bits, &[min, mask]), Ok(0));
        assert_eq!(BvOp::Shl.eval(bits, &[1, bits as u128]), Ok(0));
        assert_eq!(BvOp::LShr.eval(bits, &[mask, bits as u128]), Ok(0));
        assert_eq!(BvOp::AShr.eval(bits, &[min, bits as u128]), Ok(mask));
        assert_eq!(BvOp::Clz.eval(bits, &[0]), Ok(bits as u128));
        assert_eq!(BvOp::Ctz.eval(bits, &[0]), Ok(bits as u128));
        assert_eq!(BvOp::Popcnt.eval(bits, &[mask]), Ok(bits as u128));
        for bit in 0..bits {
            assert_eq!(
                BvOp::Clz.eval(bits, &[1 << bit]),
                Ok((bits - bit - 1) as u128)
            );
            assert_eq!(BvOp::Ctz.eval(bits, &[1 << bit]), Ok(bit as u128));
            assert_eq!(BvOp::Popcnt.eval(bits, &[1 << bit]), Ok(1));
        }
    }
}

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
