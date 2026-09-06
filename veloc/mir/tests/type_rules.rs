use veloc_mir::opcode::TypeError;
use veloc_mir::types::TypeBits;
use veloc_mir::{Opcode, Type, TypeSize};

#[test]
fn logical_bits_are_independent_of_byte_storage() {
    assert_eq!(Type::BOOL.element_bits(), Some(1));
    assert_eq!(Type::BOOL.bit_size(), Some(TypeBits::Fixed(1)));
    assert_eq!(Type::BOOL.storage_size(), TypeSize::Fixed(1));

    for scalable in [false, true] {
        let mask = Type::new_mask(4, scalable).unwrap();
        let vector = Type::I8
            .as_scalar()
            .unwrap()
            .vector(4, scalable)
            .unwrap()
            .as_type();
        assert_eq!(mask.element_bits(), Some(1));
        assert_eq!(vector.element_bits(), Some(8));
        assert_eq!(mask.min_bit_width(), Some(4));
        assert_eq!(vector.min_bit_width(), Some(32));
        assert_eq!(mask.storage_size(), vector.storage_size());
        assert_ne!(mask.bit_size(), vector.bit_size());
    }

    assert_eq!(Type::PTR.element_bits(), None);
    assert_eq!(Type::PTR.bit_size(), None);
    assert_eq!(Type::PTR.storage_size(), TypeSize::TargetDependent);
}

#[test]
fn mask_widening_uses_element_bits_and_preserves_shape() {
    let extend = Opcode::ExtendU;
    assert!(extend.validate_types(&[Type::BOOL], &[Type::I8]).is_ok());

    for scalable in [false, true] {
        let mask = Type::new_mask(4, scalable).unwrap();
        let widened = Type::I8
            .as_scalar()
            .unwrap()
            .vector(4, scalable)
            .unwrap()
            .as_type();
        let wrong_lanes = Type::I8
            .as_scalar()
            .unwrap()
            .vector(8, scalable)
            .unwrap()
            .as_type();
        let wrong_scale = Type::I8
            .as_scalar()
            .unwrap()
            .vector(4, !scalable)
            .unwrap()
            .as_type();
        assert!(extend.validate_types(&[mask], &[widened]).is_ok());
        assert!(extend.validate_types(&[mask], &[wrong_lanes]).is_err());
        assert!(extend.validate_types(&[mask], &[wrong_scale]).is_err());
    }
}

#[test]
fn integer_conversions_compare_lane_widths() {
    for scalable in [false, true] {
        let narrow = Type::I8
            .as_scalar()
            .unwrap()
            .vector(4, scalable)
            .unwrap()
            .as_type();
        let wide = Type::I32
            .as_scalar()
            .unwrap()
            .vector(4, scalable)
            .unwrap()
            .as_type();
        for opcode in [Opcode::ExtendS, Opcode::ExtendU] {
            let scheme = opcode;
            assert!(scheme.validate_types(&[narrow], &[wide]).is_ok());
            assert!(scheme.validate_types(&[wide], &[narrow]).is_err());
            assert!(scheme.validate_types(&[wide], &[wide]).is_err());
        }
        let wrap = Opcode::Wrap;
        assert!(wrap.validate_types(&[wide], &[narrow]).is_ok());
        assert!(wrap.validate_types(&[narrow], &[wide]).is_err());
    }
}

#[test]
fn bitcasts_require_equal_size_expressions() {
    let bitcast = Opcode::Reinterpret;
    let fixed = Type::I32X4;
    let scalable = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, true)
        .unwrap()
        .as_type();
    assert_eq!(fixed.min_bit_width(), scalable.min_bit_width());
    assert_ne!(fixed.bit_size(), scalable.bit_size());
    assert_eq!(fixed.bit_size().and_then(TypeBits::fixed_bits), Some(128));
    assert_eq!(scalable.bit_size().and_then(TypeBits::fixed_bits), None);
    assert!(bitcast.validate_types(&[fixed], &[scalable]).is_err());
    assert!(bitcast.validate_types(&[scalable], &[fixed]).is_err());

    for scalable in [false, true] {
        let from = Type::I32
            .as_scalar()
            .unwrap()
            .vector(4, scalable)
            .unwrap()
            .as_type();
        let same_bits = Type::I64
            .as_scalar()
            .unwrap()
            .vector(2, scalable)
            .unwrap()
            .as_type();
        let more_bits = Type::I64
            .as_scalar()
            .unwrap()
            .vector(4, scalable)
            .unwrap()
            .as_type();
        assert!(bitcast.validate_types(&[from], &[same_bits]).is_ok());
        assert!(bitcast.validate_types(&[from], &[more_bits]).is_err());
        assert!(bitcast.validate_types(&[from], &[from]).is_err());
    }
}

#[test]
fn validation_rejects_conflicting_bindings() {
    let scheme = Opcode::IAdd;
    let operands = [Type::I32, Type::I64];
    let checked_error = scheme.validate_types(&operands, &[Type::I32]).unwrap_err();
    assert!(matches!(
        checked_error,
        TypeError::Pattern {
            results: false,
            index: 1,
            ..
        }
    ));
    assert!(
        scheme
            .validate_types(&[Type::I32, Type::I32], &[Type::I32])
            .is_ok()
    );
}

#[test]
fn validation_checks_operand_arity_and_classes() {
    let scheme = Opcode::IAdd;
    for operands in [
        &[Type::I32][..],
        &[Type::I32, Type::I32, Type::I32][..],
        &[Type::F32, Type::F32][..],
        &[Type::I32, Type::I64][..],
    ] {
        assert!(scheme.validate_types(operands, &[Type::I32]).is_err());
    }
}
