use veloc_semantics::{
    Error, Expr, Function, IntPredicate, Outcome, Sort, Trap, Value, equivalence_query,
};

#[test]
fn traps_are_ordered_outcomes_and_guard_only_inputs_are_validated() {
    let sort = Sort::bv(8).unwrap();
    let x = Expr::input(0, sort);
    let guard = x
        .compare(&Expr::bv(8, 0).unwrap(), IntPredicate::new(false, 2))
        .unwrap();
    let function = Function::with_traps(
        [sort],
        [Expr::bv(8, 42).unwrap()],
        [
            (guard, Trap::DivisionByZero),
            (Expr::bool(true), Trap::IntegerOverflow),
        ],
    )
    .unwrap();
    assert_eq!(
        function.execute(&[Value::Bv(0)]),
        Ok(Outcome::Trap(Trap::DivisionByZero))
    );
    assert_eq!(
        function.execute(&[Value::Bv(1)]),
        Ok(Outcome::Trap(Trap::IntegerOverflow))
    );
    assert_eq!(
        function.eval(&[Value::Bv(0)]),
        Err(Error::Trapped(Trap::DivisionByZero))
    );
    assert!(Function::with_traps([sort], [x.clone()], [(x, Trap::DivisionByZero)]).is_err());
    assert!(
        Function::with_traps(
            [],
            [Expr::bv(8, 0).unwrap()],
            [(Expr::input(0, Sort::Bool), Trap::DivisionByZero)]
        )
        .is_err()
    );
}

#[test]
fn query_compares_traps_and_observes_values_only_on_normal_returns() {
    let make = |value, trap| {
        Function::with_traps(
            [],
            [Expr::bv(8, value).unwrap()],
            [(Expr::bool(true), trap)],
        )
        .unwrap()
    };
    let source = make(0, Trap::DivisionByZero);
    let target = make(42, Trap::DivisionByZero);
    assert_eq!(source.execute(&[]), target.execute(&[]));
    let query = equivalence_query(&source, &target).unwrap();
    assert!(query.contains("(not (= s_trap t_trap))"));
    assert!(query.contains("(and (= s_trap (_ bv0 8))"));
    assert_ne!(
        source.execute(&[]),
        make(0, Trap::IntegerOverflow).execute(&[])
    );
}
