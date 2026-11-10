use veloc_mir::constant::ScalarConst;
use veloc_mir::{InstFields, Opcode, Type};
use veloc_optimizer::evaluate::{Fold, can_fold, fold};

#[test]
fn folding_returns_constants_or_original_operands() {
    // PtrOffset has a local rule but no scalar constant evaluator.
    assert!(can_fold(Opcode::PtrOffset));
    let fields = InstFields::from_opcode(Opcode::IAdd).unwrap();
    let args = [0usize, 1];
    assert!(can_fold(Opcode::IAdd));
    assert_eq!(
        fold(&fields, &args, &[Type::I32], |i| {
            (i == 1).then(|| ScalarConst::from(0i32))
        })
        .unwrap()
        .as_slice(),
        &[Fold::Operand(0)]
    );
    assert_eq!(
        fold(&fields, &args, &[Type::I32], |i| {
            Some(ScalarConst::from([2i32, 3][i]))
        })
        .unwrap()
        .as_slice(),
        &[Fold::Constant(ScalarConst::from(5i32))]
    );
    assert!(fold(&fields, &args, &[Type::I32], |_| None).is_none());
}

#[test]
fn operand_folds_support_results_outside_scalar_constants() {
    let vector = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();
    assert!(ScalarConst::from_bits(vector, 0).is_none());
    assert!(can_fold(Opcode::Select));
    let fields = InstFields::from_opcode(Opcode::Select).unwrap();
    assert_eq!(
        fold(&fields, &[0usize, 1, 2], &[vector], |i| {
            (i == 0).then(|| ScalarConst::from(true))
        })
        .unwrap()
        .as_slice(),
        &[Fold::Operand(1)]
    );
    assert_eq!(
        fold(&fields, &[0usize, 1, 1], &[vector], |_| None)
            .unwrap()
            .as_slice(),
        &[Fold::Operand(1)]
    );
    // Opcode support does not promise that a specific type can be folded.
    assert!(can_fold(Opcode::INeg));
    let fields = InstFields::from_opcode(Opcode::INeg).unwrap();
    assert!(fold(&fields, &[0usize], &[vector], |_| None).is_none());
    assert!(!can_fold(Opcode::FAdd));
    assert!(!can_fold(Opcode::Return));
}

#[test]
fn scalar_only_folding_preserves_traps_and_multiple_results() {
    let fields = InstFields::from_opcode(Opcode::IDivS).unwrap();
    assert!(can_fold(Opcode::IDivS));
    assert!(
        fold(&fields, &[0usize, 1], &[Type::I32], |i| {
            Some(ScalarConst::from([1i32, 0][i]))
        })
        .is_none()
    );
    let fields = InstFields::from_opcode(Opcode::IAddWithOverflow).unwrap();
    let results = [Type::I32, Type::BOOL];
    assert!(can_fold(Opcode::IAddWithOverflow));
    assert_eq!(
        fold(&fields, &[0usize, 1], &results, |i| {
            Some(ScalarConst::from([i32::MAX, 1][i]))
        })
        .unwrap()
        .as_slice(),
        &[
            Fold::Constant(ScalarConst::from(i32::MIN)),
            Fold::Constant(ScalarConst::from(true)),
        ]
    );
}
