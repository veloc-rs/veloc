use veloc_mir::constant::Constant;
use veloc_mir::semantics::{Outcome, Sort, Trap, Value};
use veloc_mir::{IntCC, Opcode, Type};

#[test]
fn division_guards_and_remainder_cover_all_integer_widths() {
    for bits in [8, 16, 32, 64] {
        let sort = Sort::bv(bits).unwrap();
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        let mask = (1u128 << bits) - 1;
        for op in [Opcode::IDivS, Opcode::IDivU, Opcode::IRemS, Opcode::IRemU] {
            let program = op.spec().semantics.unwrap();
            assert_eq!(program.primitive(), None);
            let function = program.instantiate(&[sort, sort], &[sort], &[]).unwrap();
            for x in [min, -7, -1, 0, 1, 7, max] {
                for y in [min, -3, -1, 0, 1, 3, max] {
                    let expected = if y == 0 {
                        Outcome::Trap(Trap::DivisionByZero)
                    } else if op == Opcode::IDivS && x == min && y == -1 {
                        Outcome::Trap(Trap::IntegerOverflow)
                    } else {
                        let value = match op {
                            Opcode::IDivS => (x / y) as u128,
                            Opcode::IRemS => (x % y) as u128,
                            Opcode::IDivU => (x as u128 & mask) / (y as u128 & mask),
                            Opcode::IRemU => (x as u128 & mask) % (y as u128 & mask),
                            _ => unreachable!(),
                        };
                        Outcome::Values(vec![Value::Bv(value & mask)])
                    };
                    assert_eq!(
                        function
                            .execute(&[Value::Bv(x as u128), Value::Bv(y as u128)])
                            .unwrap(),
                        expected,
                        "{op:?} i{bits}: {x}, {y}"
                    );
                }
            }
        }
    }
}

#[test]
fn shifts_rotations_counts_and_eqz_fold_from_definitions() {
    for bits in [8, 16, 32, 64] {
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        let mask = (1u128 << bits) - 1;
        let constant = |v: i128| match bits {
            8 => Constant::I8(v as i8),
            16 => Constant::I16(v as i16),
            32 => Constant::I32(v as i32),
            64 => Constant::I64(v as i64),
            _ => unreachable!(),
        };
        for x in [min, -7, -1, 0, 1, 7, max] {
            let raw = x as u128 & mask;
            for amount in [0, 1, bits - 1, bits, bits + 1, bits * 2 + 1, mask as u32] {
                let k = amount % bits;
                for (op, expected) in [
                    (Opcode::IShl, raw << k),
                    (Opcode::IShrU, raw >> k),
                    (Opcode::IShrS, (x >> k) as u128),
                    (Opcode::IRotl, (raw << k) | (raw >> (bits - k))),
                    (Opcode::IRotr, (raw >> k) | (raw << (bits - k))),
                ] {
                    assert_eq!(
                        constant(x).binary_op(constant(amount as i128), op),
                        Some(constant((expected & mask) as i128)),
                        "{op:?} i{bits}: {x}, {amount}"
                    );
                }
            }
            for (op, expected) in [
                (Opcode::IClz, raw.leading_zeros() - (128 - bits)),
                (Opcode::ICtz, raw.trailing_zeros().min(bits)),
                (Opcode::IPopcnt, raw.count_ones()),
            ] {
                assert_eq!(constant(x).unary_op(op), Some(constant(expected as i128)));
            }
            assert_eq!(
                constant(x).unary_op(Opcode::IEqz),
                Some(Constant::Bool(x == 0))
            );
        }
        assert_eq!(constant(min).binary_op(constant(-1), Opcode::IDivS), None);
        assert_eq!(
            constant(min).binary_op(constant(-1), Opcode::IRemS),
            Some(constant(0))
        );
        assert_eq!(constant(1).binary_op(constant(0), Opcode::IDivU), None);
    }
}

#[test]
fn signed_overflow_matches_widened_arithmetic_at_every_mir_integer_width() {
    let program = Opcode::IAddWithOverflow.spec().semantics.unwrap();
    for bits in [8, 16, 32, 64] {
        let sort = Sort::bv(bits).unwrap();
        let function = program
            .instantiate(&[sort, sort], &[sort, Sort::Bool], &[])
            .unwrap();
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        let mask = (1u128 << bits) - 1;
        for lhs in [min, min + 1, -1, 0, 1, max - 1, max] {
            for rhs in [min, min + 1, -1, 0, 1, max - 1, max] {
                let sum = lhs + rhs;
                assert_eq!(
                    function
                        .eval_all(&[Value::Bv(lhs as u128), Value::Bv(rhs as u128)])
                        .unwrap(),
                    [
                        Value::Bv(sum as u128 & mask),
                        Value::Bool(sum < min || sum > max)
                    ]
                );
            }
        }
    }
}

#[test]
fn generated_integer_comparisons_match_rust_for_all_i8_pairs() {
    let kinds = [
        IntCC::Eq,
        IntCC::Ne,
        IntCC::LtS,
        IntCC::LtU,
        IntCC::GtS,
        IntCC::GtU,
        IntCC::LeS,
        IntCC::LeU,
        IntCC::GeS,
        IntCC::GeU,
    ];
    let sort = Sort::bv(8).unwrap();
    for cc in kinds {
        let function = Opcode::Icmp
            .spec()
            .semantics
            .unwrap()
            .instantiate(&[sort, sort], &[Sort::Bool], &[cc.predicate()])
            .unwrap();
        for lhs in i8::MIN..=i8::MAX {
            for rhs in i8::MIN..=i8::MAX {
                let expected = match cc {
                    IntCC::Eq => lhs == rhs,
                    IntCC::Ne => lhs != rhs,
                    IntCC::LtS => lhs < rhs,
                    IntCC::LeS => lhs <= rhs,
                    IntCC::GtS => lhs > rhs,
                    IntCC::GeS => lhs >= rhs,
                    IntCC::LtU => (lhs as u8) < rhs as u8,
                    IntCC::LeU => (lhs as u8) <= rhs as u8,
                    IntCC::GtU => (lhs as u8) > rhs as u8,
                    IntCC::GeU => (lhs as u8) >= rhs as u8,
                };
                assert_eq!(
                    function
                        .eval(&[Value::Bv(lhs as u128), Value::Bv(rhs as u128)])
                        .unwrap(),
                    Value::Bool(expected)
                );
            }
        }
    }
}

#[test]
fn mixed_width_and_boolean_constants_use_typed_results() {
    for (op, input, ty, expected) in [
        (
            Opcode::ExtendS,
            Constant::I8(-1),
            Type::I64,
            Constant::I64(-1),
        ),
        (
            Opcode::ExtendU,
            Constant::I8(-1),
            Type::I64,
            Constant::I64(255),
        ),
        (Opcode::Wrap, Constant::I64(511), Type::I8, Constant::I8(-1)),
        (
            Opcode::ExtendU,
            Constant::Bool(true),
            Type::I32,
            Constant::I32(1),
        ),
    ] {
        assert_eq!(
            Constant::evaluate(op, &[input], &[ty], &[]),
            Some(vec![expected])
        );
    }
    assert_eq!(
        Constant::Bool(true).binary_op(Constant::Bool(false), Opcode::IAnd),
        Some(Constant::Bool(false))
    );
    assert_eq!(
        Constant::evaluate(Opcode::ExtendS, &[Constant::I64(1)], &[Type::I8], &[]),
        None
    );
    assert_eq!(
        Constant::evaluate(Opcode::Wrap, &[Constant::I8(1)], &[Type::I64], &[]),
        None
    );
    assert_eq!(
        Constant::evaluate(
            Opcode::IAddWithOverflow,
            &[Constant::I8(127), Constant::I8(1)],
            &[Type::I8, Type::BOOL],
            &[]
        ),
        Some(vec![Constant::I8(-128), Constant::Bool(true)])
    );
}
