//! Shared type encoding, checked views, and signature interning.
use veloc_types::{
    CallConv, CallableKind, ScalarType, SigId, Signature, SignatureError, Signatures, Type,
    TypeInfo,
};

#[test]
fn shared_encoding_and_checked_views_roundtrip() {
    assert_eq!(core::mem::size_of::<Type>(), 8);
    assert_eq!(
        core::mem::size_of::<ScalarType>(),
        core::mem::size_of::<Type>()
    );
    assert_eq!(
        core::mem::align_of::<ScalarType>(),
        core::mem::align_of::<Type>()
    );
    let mut valid = 0;
    for raw in 0..=u16::MAX {
        let Some(ty) = Type::from_raw(raw) else {
            continue;
        };
        valid += 1;
        assert_eq!(ty.to_raw(), raw);
        assert!(ty.is_valid());
        assert_eq!(ty.is_scalar(), ty.as_scalar().is_some());
        if let Some(vector) = ty.as_vector() {
            let (lanes, scalable) = vector.shape();
            assert_eq!(vector.element_type().vector(lanes, scalable), Some(vector));
            assert_eq!(ty.shape(), Some((lanes, scalable)));
            assert_eq!(ty.element(), Some(vector.element_type().element()));
        } else {
            let scalar = ty.as_scalar().unwrap();
            assert_eq!(scalar.as_type(), ty);
            assert_eq!(ScalarType::from_name(scalar.name()), Some(scalar));
            assert_eq!(Type::from_scalar_name(scalar.name()), Some(ty));
            assert_eq!(Type::from_scalar_code(scalar.code()), Some(ty));
            assert_eq!(ScalarType::from_element(scalar.element()), Some(scalar));
        }
    }
    assert_eq!(valid, 8 + 7 * 15 * 2);
    assert_eq!(Type::from_scalar_name("i32<4>"), None);
    for raw in 0..=u8::MAX {
        assert_eq!(ScalarType::from_code(raw).is_some(), (1..=8).contains(&raw));
    }
    for kind in [
        CallableKind::Local,
        CallableKind::Owned,
        CallableKind::Shared,
    ] {
        let ty = Type::callable(SigId(u32::MAX), kind);
        assert_eq!(ty.as_callable(), Some((SigId(u32::MAX), kind)));
        assert!(ty.is_valid());
        assert_eq!(ty.element(), None);
        assert_eq!(ty.bit_size(), None);
        assert_eq!(ty.shape(), None);
        assert!(ty.as_scalar().is_none() && ty.as_vector().is_none());
    }
}

const CC: CallConv = CallConv::SystemV;

#[test]
fn interning_preserves_payload_and_parameter_result_boundary() {
    let mut pool = Signatures::default();
    let signature = Signature::new([Type::I32], [Type::I64], CC);
    let payload = signature.types().as_ptr();
    let id = pool.insert(signature);
    assert_eq!(pool[id].types().as_ptr(), payload);
    assert_eq!(pool.intern(&[Type::I32], &[Type::I64], CC), id);
    assert_ne!(pool.intern(&[Type::I32, Type::I64], &[], CC), id);
    // Rehashing must use the same identity as borrowed lookup.
    for count in 0..1000 {
        let params = vec![Type::I32; count];
        let id = pool.intern(&params, &[], CC);
        assert_eq!(pool.insert(Signature::new(&params, [], CC)), id);
    }
    assert_eq!(pool.intern(&[Type::I32], &[Type::I64], CC), id);
    assert_eq!(pool[id].types().as_ptr(), payload);
    let mut clone = pool.clone();
    assert_eq!(clone.intern(&[Type::I32], &[Type::I64], CC), id);
    assert_eq!(pool[id].params(), &[Type::I32]);
    assert_eq!(pool[id].returns(), &[Type::I64]);
}

#[test]
fn import_remaps_nested_signatures_and_rejects_invalid_graphs_atomically() {
    let mut source = Signatures::default();
    // Forward reference: import must process the leaf before the outer signature.
    let outer = source.intern(&[Type::callable(SigId(1), CallableKind::Owned)], &[], CC);
    let leaf = source.intern(&[Type::I32], &[Type::I64], CC);
    assert_eq!(leaf, SigId(1));
    let mut target = Signatures::default();
    let canonical_leaf = target.intern(&[Type::I32], &[Type::I64], CC);
    let canonical_outer = target.intern(
        &[Type::callable(canonical_leaf, CallableKind::Owned)],
        &[],
        CC,
    );
    let ids = target.import(&source).unwrap();
    assert_eq!(ids[leaf.0 as usize], canonical_leaf);
    assert_eq!(ids[outer.0 as usize], canonical_outer);
    assert_eq!(target.len(), 2);
    let shared = target.intern(
        &[Type::callable(canonical_leaf, CallableKind::Shared)],
        &[],
        CC,
    );
    assert_ne!(shared, canonical_outer);
    let different = target.intern(&[Type::I64], &[Type::I32], CC);
    assert_ne!(different, canonical_leaf);

    for (reference, expected) in [
        (
            SigId(9),
            SignatureError::Unknown {
                source: SigId(0),
                target: SigId(9),
            },
        ),
        (
            SigId(0),
            SignatureError::Cycle {
                source: SigId(0),
                target: SigId(0),
            },
        ),
    ] {
        let mut bad = Signatures::default();
        bad.intern(&[Type::callable(reference, CallableKind::Shared)], &[], CC);
        let before = target.len();
        assert_eq!(target.import(&bad), Err(expected));
        assert_eq!(target.len(), before);
    }
}

#[test]
fn comparison_semantics_and_transforms() {
    use veloc_types::{FloatCC as F, IntCC as I};
    let integers = &[
        I::Eq,
        I::Ne,
        I::LtS,
        I::LtU,
        I::GtS,
        I::GtU,
        I::LeS,
        I::LeU,
        I::GeS,
        I::GeU,
    ];
    let floats = &[F::Eq, F::Ne, F::Lt, F::Gt, F::Le, F::Ge];
    for &cc in integers {
        assert_eq!(I::from_mnemonic(cc.mnemonic()), Some(cc));
        assert_eq!(cc.swap().swap(), cc);
        assert_eq!(cc.complement().complement(), cc);
        for x in 0..=255u128 {
            for y in 0..=255u128 {
                let actual = cc.test(8, x, y);
                let expected = match cc {
                    I::Eq => x == y,
                    I::Ne => x != y,
                    I::LtS => (x as i8) < (y as i8),
                    I::LtU => x < y,
                    I::GtS => (x as i8) > (y as i8),
                    I::GtU => x > y,
                    I::LeS => (x as i8) <= (y as i8),
                    I::LeU => x <= y,
                    I::GeS => (x as i8) >= (y as i8),
                    I::GeU => x >= y,
                };
                assert_eq!(actual, expected);
                assert_eq!(actual, cc.swap().test(8, y, x));
                assert_ne!(actual, cc.complement().test(8, x, y));
            }
        }
    }
    for width in [1, 8, 16, 32, 64, 128] {
        let sign = 1u128 << (width - 1);
        assert!(I::LtS.test(width, sign, 0));
        assert!(I::GtU.test(width, sign, 0));
        assert!(I::Eq.test(width, u128::MAX, u128::MAX >> (128 - width)));
    }
    for &cc in floats {
        assert_eq!(F::from_mnemonic(cc.mnemonic()), Some(cc));
        assert_eq!(cc.swap().swap(), cc);
        if let Some(complement) = cc.complement() {
            assert_eq!(cc.outcomes() ^ complement.outcomes(), 15);
            assert_eq!(complement.complement(), Some(cc));
        } else {
            assert!(
                !floats
                    .iter()
                    .any(|other| other.outcomes() == cc.outcomes() ^ 15)
            );
        }
        assert_eq!((cc.outcomes() ^ cc.complement_ordered().outcomes()) & 7, 7);
        for x in [
            f64::NAN,
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::INFINITY,
        ] {
            for y in [f64::NAN, -1.0, 0.0, 1.0] {
                let outcome = if x.is_nan() || y.is_nan() {
                    8
                } else if x < y {
                    1
                } else if x == y {
                    2
                } else {
                    4
                };
                let expected = match cc {
                    F::Eq => x == y,
                    F::Ne => x != y,
                    F::Lt => x < y,
                    F::Gt => x > y,
                    F::Le => x <= y,
                    F::Ge => x >= y,
                };
                assert_eq!(cc.outcomes() & outcome != 0, expected);
            }
        }
    }
}
