use veloc_ir::opspec::{
    ResultTypes, TypeClass, TypePattern, TypeRelation, TypeScheme, TypeSchemeError, TypeSlot,
};
use veloc_ir::types::TypeBits;
use veloc_ir::{Opcode, ScalarType, Type, TypeSize};

#[test]
fn logical_bits_are_independent_of_byte_storage() {
    assert_eq!(Type::BOOL.element_bits(), Some(1));
    assert_eq!(Type::BOOL.bit_size(), Some(TypeBits::Fixed(1)));
    assert_eq!(Type::BOOL.storage_size(), TypeSize::Fixed(1));

    for scalable in [false, true] {
        let mask = Type::new_mask(4, scalable).unwrap();
        let vector = Type::new_vector(ScalarType::I8, 4, scalable).unwrap();
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
    let extend = Opcode::ExtendU.spec().type_scheme;
    assert!(extend.validate(&[Type::BOOL], &[Type::I8]).is_ok());

    for scalable in [false, true] {
        let mask = Type::new_mask(4, scalable).unwrap();
        let widened = Type::new_vector(ScalarType::I8, 4, scalable).unwrap();
        let wrong_lanes = Type::new_vector(ScalarType::I8, 8, scalable).unwrap();
        let wrong_scale = Type::new_vector(ScalarType::I8, 4, !scalable).unwrap();
        assert!(extend.validate(&[mask], &[widened]).is_ok());
        assert!(extend.validate(&[mask], &[wrong_lanes]).is_err());
        assert!(extend.validate(&[mask], &[wrong_scale]).is_err());
    }
}

#[test]
fn integer_conversions_compare_lane_widths() {
    for scalable in [false, true] {
        let narrow = Type::new_vector(ScalarType::I8, 4, scalable).unwrap();
        let wide = Type::new_vector(ScalarType::I32, 4, scalable).unwrap();
        for opcode in [Opcode::ExtendS, Opcode::ExtendU] {
            let scheme = opcode.spec().type_scheme;
            assert!(scheme.validate(&[narrow], &[wide]).is_ok());
            assert!(scheme.validate(&[wide], &[narrow]).is_err());
            assert!(scheme.validate(&[wide], &[wide]).is_err());
        }
        let wrap = Opcode::Wrap.spec().type_scheme;
        assert!(wrap.validate(&[wide], &[narrow]).is_ok());
        assert!(wrap.validate(&[narrow], &[wide]).is_err());
    }
}

#[test]
fn bitcasts_require_equal_size_expressions() {
    let bitcast = Opcode::Reinterpret.spec().type_scheme;
    let fixed = Type::I32X4;
    let scalable = Type::new_vector(ScalarType::I32, 4, true).unwrap();
    assert_eq!(fixed.min_bit_width(), scalable.min_bit_width());
    assert_ne!(fixed.bit_size(), scalable.bit_size());
    assert_eq!(fixed.bit_size().and_then(TypeBits::fixed_bits), Some(128));
    assert_eq!(scalable.bit_size().and_then(TypeBits::fixed_bits), None);
    assert!(bitcast.validate(&[fixed], &[scalable]).is_err());
    assert!(bitcast.validate(&[scalable], &[fixed]).is_err());

    for scalable in [false, true] {
        let from = Type::new_vector(ScalarType::I32, 4, scalable).unwrap();
        let same_bits = Type::new_vector(ScalarType::I64, 2, scalable).unwrap();
        let more_bits = Type::new_vector(ScalarType::I64, 4, scalable).unwrap();
        assert!(bitcast.validate(&[from], &[same_bits]).is_ok());
        assert!(bitcast.validate(&[from], &[more_bits]).is_err());
        assert!(bitcast.validate(&[from], &[from]).is_err());
    }
}

#[test]
fn inference_and_validation_reject_conflicting_bindings() {
    const OPERANDS: &[TypePattern] = &[
        TypePattern::Bind(0, TypeClass::Integer),
        TypePattern::Bind(0, TypeClass::Integer),
    ];
    const RESULTS: &[TypePattern] = &[TypePattern::Same(0)];
    let scheme = TypeScheme::fixed(OPERANDS, RESULTS);
    let operands = [Type::I32, Type::I64];
    let inferred_error = scheme.infer_results(&operands).unwrap_err();
    let checked_error = scheme.validate(&operands, &[Type::I32]).unwrap_err();
    assert_eq!(inferred_error, checked_error);
    assert!(matches!(
        inferred_error,
        TypeSchemeError::Pattern {
            results: false,
            index: 1,
            ..
        }
    ));
    assert_eq!(
        scheme.infer_results(&[Type::I32, Type::I32]),
        Ok(ResultTypes::Inferred(smallvec::smallvec![Type::I32]))
    );
}

#[test]
fn type_schemes_support_more_than_four_variables() {
    const OPERANDS: &[TypePattern] = &[
        TypePattern::Bind(0, TypeClass::Any),
        TypePattern::Bind(1, TypeClass::Any),
        TypePattern::Bind(2, TypeClass::Any),
        TypePattern::Bind(3, TypeClass::Any),
        TypePattern::Bind(4, TypeClass::Any),
    ];
    const RESULTS: &[TypePattern] = &[TypePattern::Same(4), TypePattern::Same(0)];
    let scheme = TypeScheme::fixed(OPERANDS, RESULTS);
    let operands = [Type::I8, Type::I16, Type::I32, Type::I64, Type::F32];
    assert_eq!(
        scheme.infer_results(&operands),
        Ok(ResultTypes::Inferred(smallvec::smallvec![
            Type::F32,
            Type::I8
        ]))
    );
    assert!(scheme.validate(&operands, &[Type::F32, Type::I8]).is_ok());
    assert!(scheme.validate(&operands, &[Type::I8, Type::F32]).is_err());
}

#[test]
fn inference_checks_operand_arity_and_classes() {
    let scheme = Opcode::IAdd.spec().type_scheme;
    for operands in [
        &[Type::I32][..],
        &[Type::I32, Type::I32, Type::I32][..],
        &[Type::F32, Type::F32][..],
        &[Type::I32, Type::I64][..],
    ] {
        assert_eq!(
            scheme.infer_results(operands).unwrap_err(),
            scheme.validate(operands, &[Type::I32]).unwrap_err()
        );
    }
}

#[test]
fn inferred_results_must_satisfy_relations() {
    const OPERANDS: &[TypePattern] = &[TypePattern::Bind(0, TypeClass::Integer)];
    const RESULTS: &[TypePattern] = &[TypePattern::Exact(Type::I8)];
    const RELATION: TypeRelation = TypeRelation::Wider {
        from: TypeSlot::Operand(0),
        to: TypeSlot::Result(0),
    };
    let scheme = TypeScheme::fixed(OPERANDS, RESULTS).with_relations(&[RELATION]);
    assert_eq!(
        scheme.infer_results(&[Type::I32]),
        Err(TypeSchemeError::Relation(RELATION))
    );
    assert_eq!(
        scheme.infer_results(&[Type::I32]).unwrap_err(),
        scheme.validate(&[Type::I32], &[Type::I8]).unwrap_err()
    );
}

#[test]
fn inferred_results_must_satisfy_bound_classes() {
    const OPERANDS: &[TypePattern] = &[TypePattern::Bind(0, TypeClass::Integer)];
    const RESULTS: &[TypePattern] = &[TypePattern::Bind(0, TypeClass::Float)];
    let scheme = TypeScheme::fixed(OPERANDS, RESULTS);
    assert_eq!(
        scheme.infer_results(&[Type::I32]).unwrap_err(),
        scheme.validate(&[Type::I32], &[Type::I32]).unwrap_err()
    );
}
