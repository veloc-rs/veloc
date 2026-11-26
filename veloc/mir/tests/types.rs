//! Generated type contracts across scalar, vector, and malformed IR.
use veloc_mir::inst::{TypeClass, TypeError};
use veloc_mir::types::TypeBits;
use veloc_mir::{Opcode, Type};

#[test]
fn logical_bits_are_independent_of_byte_storage() {
    assert_eq!(Type::BOOL.element_bits(), Some(1));
    assert_eq!(Type::BOOL.bit_size(), Some(TypeBits::Fixed(1)));

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
        assert_ne!(mask.bit_size(), vector.bit_size());
    }

    assert_eq!(Type::PTR.element_bits(), None);
    assert_eq!(Type::PTR.bit_size(), None);
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
    let fixed = veloc_mir::Type::I32X4;
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

#[test]
fn operand_contract_errors_are_independent_of_result_types() {
    let types = [
        Type::INVALID,
        Type::BOOL,
        Type::PTR,
        Type::I8,
        Type::I16,
        Type::I32,
        Type::I64,
        Type::F32,
        Type::F64,
        veloc_mir::Type::I32X4,
        veloc_mir::Type::I64X2,
        Type::I8
            .as_scalar()
            .unwrap()
            .vector(4, false)
            .unwrap()
            .as_type(),
        Type::I16
            .as_scalar()
            .unwrap()
            .vector(4, false)
            .unwrap()
            .as_type(),
        Type::I32
            .as_scalar()
            .unwrap()
            .vector(4, true)
            .unwrap()
            .as_type(),
        Type::I64
            .as_scalar()
            .unwrap()
            .vector(4, true)
            .unwrap()
            .as_type(),
        Type::new_mask(4, false).unwrap(),
        Type::new_mask(4, true).unwrap(),
    ];
    for &op in Opcode::ALL {
        // Cover invalid encodings, scalar/vector shapes and wrong arities.
        for x in types {
            for y in types {
                let storage = [x, y, x];
                for len in 0..=3 {
                    let operands = &storage[..len];
                    if let Err(
                        error @ (TypeError::Arity { results: false, .. }
                        | TypeError::Pattern { results: false, .. }),
                    ) = op.validate_types(operands, &[])
                    {
                        for result in types {
                            for results in [&[][..], &[result][..], &[result, x][..]] {
                                assert_eq!(
                                    op.validate_types(operands, results),
                                    Err(error),
                                    "{op:?} {operands:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn variadic_operands_validate_the_known_prefix() {
    let branch = Opcode::Br;
    assert!(branch.validate_types(&[Type::BOOL, Type::I64], &[]).is_ok());
    assert!(branch.validate_types(&[Type::I32, Type::I64], &[]).is_err());
    assert!(branch.validate_types(&[], &[]).is_err());
    let call = Opcode::CallIndirect;
    assert!(
        call.validate_types(&[Type::PTR, Type::I32], &[Type::I64])
            .is_ok()
    );
    assert!(
        call.validate_types(&[Type::I64, Type::I32], &[Type::I64])
            .is_err()
    );
    assert!(call.validate_types(&[], &[]).is_err());
}

#[test]
fn inline_type_sets_preserve_membership_and_shared_bindings() {
    for raw in 0..=u16::MAX {
        let Some(ty) = Type::from_raw(raw) else {
            continue;
        };
        for op in [Opcode::IAnd, Opcode::IOr, Opcode::IXor] {
            assert_eq!(
                op.validate_types(&[ty, ty], &[ty]).is_ok(),
                ty.is_integer() || ty == Type::BOOL || ty.is_predicate(),
                "{op:?}: {ty}"
            );
        }
        assert_eq!(
            Opcode::Icmp
                .validate_types(&[ty, ty], &[Type::BOOL])
                .is_ok(),
            ty.is_scalar() && (ty.is_integer() || ty.is_ptr()),
            "{ty}"
        );
        assert_eq!(
            Opcode::Gather
                .validate_types(&[Type::PTR, ty], &[ty])
                .is_ok(),
            ty.is_integer() && ty.is_vector(),
            "{ty}"
        );
    }
    for op in [Opcode::IAnd, Opcode::Icmp, Opcode::Gather] {
        assert!(
            op.validate_types(&[Type::INVALID, Type::INVALID], &[Type::INVALID])
                .is_err()
        );
    }
    let scheme = Opcode::IAnd;
    assert!(
        scheme
            .validate_types(&[Type::I32, Type::I32], &[Type::I32])
            .is_ok()
    );
    assert!(
        scheme
            .validate_types(&[Type::BOOL, Type::BOOL], &[Type::BOOL])
            .is_ok()
    );
    assert!(
        scheme
            .validate_types(&[Type::I32, Type::I64], &[Type::I32])
            .is_err()
    );
    assert!(
        scheme
            .validate_types(&[Type::I32, Type::I32], &[Type::BOOL])
            .is_err()
    );
}

#[test]
fn generated_classes_preserve_membership_for_every_valid_encoding() {
    for raw in 0..=u16::MAX {
        let Some(ty) = Type::from_raw(raw) else {
            continue;
        };
        for (class, expected) in [
            (TypeClass::Any, true),
            (TypeClass::Scalar, ty.is_scalar() && !ty.is_ptr()),
            (TypeClass::ScalarInteger, ty.is_scalar() && ty.is_integer()),
            (TypeClass::ScalarFloat, ty.is_scalar() && ty.is_float()),
            (TypeClass::Integer, ty.is_integer()),
            (TypeClass::Float, ty.is_float()),
            (TypeClass::Number, ty.is_integer() || ty.is_float()),
            (TypeClass::Vector, ty.is_vector()),
        ] {
            assert_eq!(class.accepts(ty), expected, "{class:?}: {ty}");
            assert!(!class.accepts(Type::INVALID), "{class:?}");
        }
    }
}
