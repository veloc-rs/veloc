mod offline {
    include!(concat!(env!("OUT_DIR"), "/semantics.rs"));
}
use veloc_mir::constant::Constant;
use veloc_mir::{IntCC, Opcode, Type};
use veloc_semantics::{Outcome, Sort, Trap, Value};

fn program(op: Opcode) -> veloc_semantics::Program<'static> {
    let specs = offline::SPECS;
    specs
        .iter()
        .find(|s| s.opcode == op)
        .expect("modeled opcode")
        .program
}

#[test]
fn offline_table_matches_runtime_metadata_without_duplicate_opcodes() {
    let specs = offline::SPECS;
    for (i, spec) in specs.iter().enumerate() {
        assert!(!specs[..i].iter().any(|s| s.opcode == spec.opcode));
        assert_eq!(
            spec.program.inputs as usize,
            match spec.opcode.spec().type_scheme.operands {
                veloc_mir::opspec::TypeList::Fixed(p) => p.len(),
                _ => unreachable!(),
            }
        );
        spec.program.validate().unwrap();
    }
    assert!(Opcode::ALL.iter().all(
        |op| !veloc_optimizer::rewrite::can_fold(*op) || specs.iter().any(|s| s.opcode == *op)
    ));
    assert!(!specs.iter().any(|s| s.opcode == Opcode::FAdd));
}

#[test]
fn division_guards_and_remainder_cover_all_integer_widths() {
    for bits in [8, 16, 32, 64] {
        let sort = Sort::bv(bits).unwrap();
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        let mask = (1u128 << bits) - 1;
        for op in [Opcode::IDivS, Opcode::IDivU, Opcode::IRemS, Opcode::IRemU] {
            let program = program(op);
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
    let program = program(Opcode::IAddWithOverflow);
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
    for cc in kinds {
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
                    Constant::I8(lhs).icmp(Constant::I8(rhs), cc),
                    Some(Constant::Bool(expected))
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
            veloc_optimizer::rewrite::evaluate(op, &[input], &[ty], &[]),
            Some(vec![expected])
        );
    }
    assert_eq!(
        Constant::Bool(true).binary_op(Constant::Bool(false), Opcode::IAnd),
        Some(Constant::Bool(false))
    );
    assert_eq!(
        veloc_optimizer::rewrite::evaluate(Opcode::ExtendS, &[Constant::I64(1)], &[Type::I8], &[]),
        None
    );
    assert_eq!(
        veloc_optimizer::rewrite::evaluate(Opcode::Wrap, &[Constant::I8(1)], &[Type::I64], &[]),
        None
    );
    assert_eq!(
        veloc_optimizer::rewrite::evaluate(
            Opcode::IAddWithOverflow,
            &[Constant::I8(127), Constant::I8(1)],
            &[Type::I8, Type::BOOL],
            &[]
        ),
        Some(vec![Constant::I8(-128), Constant::Bool(true)])
    );
}

/// Offline differential check: every legal scalar signature is checked against
/// the graph interpreter, including every MIR IntCC predicate.
/// Samples exercise boundaries and deterministic random values, not an SMT proof.
#[test]
fn generated_evaluators_match_graphs_for_all_scalar_signatures() {
    const TYPES: [Type; 5] = [Type::I8, Type::I16, Type::I32, Type::I64, Type::BOOL];
    fn tuples(n: usize) -> Vec<Vec<Type>> {
        if n == 0 {
            return vec![vec![]];
        }
        tuples(n - 1)
            .into_iter()
            .flat_map(|prefix| {
                TYPES.map(|ty| {
                    let mut row = prefix.clone();
                    row.push(ty);
                    row
                })
            })
            .collect()
    }
    fn constant(ty: Type, bits: u128) -> Constant {
        match ty {
            Type::I8 => Constant::I8(bits as i8),
            Type::I16 => Constant::I16(bits as i16),
            Type::I32 => Constant::I32(bits as i32),
            Type::I64 => Constant::I64(bits as i64),
            Type::BOOL => Constant::Bool(bits & 1 != 0),
            _ => unreachable!(),
        }
    }
    let sort = |ty: Type| {
        if ty == Type::BOOL {
            Sort::Bool
        } else {
            Sort::bv(ty.min_bit_width().unwrap() as u16).unwrap()
        }
    };
    let predicates = [
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
    let mut seed = 0x9183_acf2_ee74_b01du64;
    let mut checked = 0;
    for &op in Opcode::ALL {
        if !veloc_optimizer::rewrite::can_fold(op) {
            continue;
        }
        let program = program(op);
        assert!(program.properties <= 1, "extend predicate product coverage");
        for signature in tuples(program.inputs as usize + program.outputs.len()) {
            let (inputs, outputs) = signature.split_at(program.inputs as usize);
            if op.validate_types(inputs, outputs).is_err() {
                continue;
            }
            let properties = if program.properties == 0 {
                vec![vec![]]
            } else {
                predicates.iter().map(|&p| vec![p]).collect()
            };
            for properties in properties {
                let function = program
                    .instantiate(
                        &inputs.iter().copied().map(sort).collect::<Vec<_>>(),
                        &outputs.iter().copied().map(sort).collect::<Vec<_>>(),
                        &properties
                            .iter()
                            .copied()
                            .map(offline::predicate)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                // Cartesian boundary cases for binary operands, followed by random samples.
                for sample in 0..96 {
                    let args = inputs
                        .iter()
                        .enumerate()
                        .map(|(i, &ty)| {
                            let width = if ty == Type::BOOL {
                                1
                            } else {
                                ty.min_bit_width().unwrap()
                            };
                            let sign = 1u128 << (width - 1);
                            let mask = (1u128 << width) - 1;
                            let edge = [
                                0,
                                1,
                                sign - 1,
                                sign,
                                mask,
                                u128::from(width),
                                u128::from(width - 1),
                                u128::from(width + 1),
                            ];
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            let bits = if sample < 64 {
                                edge[(sample / 8usize.pow(i as u32)) % 8]
                            } else {
                                seed as u128
                            };
                            constant(ty, bits)
                        })
                        .collect::<Vec<_>>();
                    let values = args
                        .iter()
                        .map(|&c| match c {
                            Constant::Bool(b) => Value::Bool(b),
                            _ => Value::Bv(c.as_i64().unwrap() as u128),
                        })
                        .collect::<Vec<_>>();
                    let expected = match function.execute(&values).unwrap() {
                        Outcome::Trap(_) => None,
                        Outcome::Values(values) => Some(
                            values
                                .into_iter()
                                .zip(outputs)
                                .map(|(v, &ty)| match v {
                                    Value::Bool(b) => Constant::Bool(b),
                                    Value::Bv(v) => constant(ty, v),
                                })
                                .collect::<Vec<_>>(),
                        ),
                    };
                    assert_eq!(
                        veloc_optimizer::rewrite::evaluate(op, &args, outputs, &properties),
                        expected,
                        "{op:?} {signature:?} {args:?} {properties:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 10_000);
}
trait Fold {
    fn binary_op(self, other: Self, op: Opcode) -> Option<Self>
    where
        Self: Sized;
    fn unary_op(self, op: Opcode) -> Option<Self>
    where
        Self: Sized;
    fn icmp(self, other: Self, cc: IntCC) -> Option<Self>
    where
        Self: Sized;
}
fn fold(op: Opcode, args: &[Constant]) -> Option<Constant> {
    let types = args.iter().map(|c| c.ty()).collect::<Vec<_>>();
    let veloc_mir::opspec::ResultTypes::Inferred(results) = op.infer_result_types(&types).ok()?
    else {
        return None;
    };
    veloc_optimizer::rewrite::evaluate(op, args, &results, &[])?
        .first()
        .copied()
}
impl Fold for Constant {
    fn binary_op(self, other: Self, op: Opcode) -> Option<Self> {
        fold(op, &[self, other])
    }
    fn unary_op(self, op: Opcode) -> Option<Self> {
        fold(op, &[self])
    }
    fn icmp(self, other: Self, cc: IntCC) -> Option<Self> {
        veloc_optimizer::rewrite::evaluate(Opcode::Icmp, &[self, other], &[Type::BOOL], &[cc])?
            .first()
            .copied()
    }
}
