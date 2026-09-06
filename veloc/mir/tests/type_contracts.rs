//! Exercise compiled contracts across every opcode, including malformed types.
use veloc_mir::opspec::TypeError;
use veloc_mir::{Opcode, Type};

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
        Type::I32X4,
        Type::I64X2,
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
