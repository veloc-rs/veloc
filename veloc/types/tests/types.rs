//! Shared type encoding, checked views, and signature interning.
use veloc_types::{
    CallConv, CallableKind, ScalarType, SigId, Signature, SignatureError, Signatures, Type,
    TypeInfo,
};

#[test]
fn shared_encoding_and_checked_views_roundtrip() {
    assert_eq!(core::mem::size_of::<Type>(), 8);
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
            assert_eq!(Type::from_scalar_code(scalar.code()), Some(ty));
            assert_eq!(ScalarType::from_element(scalar.element()), Some(scalar));
        }
    }
    assert_eq!(valid, 8 + 7 * 15 * 2);
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
