use veloc_types::{CallableKind, ScalarType, SigId, Type};

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
