use veloc_semantics::{BvOp, Error, Expr, Function, Sort, Value, equivalence_query};

fn split_add(width: u16, include_carry: bool) -> (Function, Function) {
    let half = width / 2;
    let sort = Sort::bv(width).unwrap();
    let x = Expr::input(0, sort);
    let y = Expr::input(1, sort);
    let source = Function::new(
        [sort, sort],
        Expr::apply(BvOp::Add, &[x.clone(), y.clone()]).unwrap(),
    )
    .unwrap();
    let xl = x.extract(half - 1, 0).unwrap();
    let yl = y.extract(half - 1, 0).unwrap();
    let xh = x.extract(width - 1, half).unwrap();
    let yh = y.extract(width - 1, half).unwrap();
    let lo = Expr::apply(BvOp::Add, &[xl.clone(), yl]).unwrap();
    let mut hi = Expr::apply(BvOp::Add, &[xh, yh]).unwrap();
    if include_carry {
        let carry = lo
            .ult(&xl)
            .unwrap()
            .bool_to_bv(1)
            .unwrap()
            .zero_extend(half)
            .unwrap();
        hi = Expr::apply(BvOp::Add, &[hi, carry]).unwrap();
    }
    let target = Function::new([sort, sort], Expr::concat(&hi, &lo).unwrap()).unwrap();
    (source, target)
}

#[test]
fn split_add_covers_carry_and_full_width_wraparound() {
    for width in [4, 8, 16, 32, 64, 128] {
        let (source, target) = split_add(width, true);
        let mask = Sort::bv(width).unwrap().width().unwrap().mask();
        let half = (1u128 << (width / 2)) - 1;
        let cases = [0, 1, half, half + 1, mask - 1, mask];
        for x in cases {
            for y in cases {
                let args = [Value::Bv(x), Value::Bv(y)];
                assert_eq!(
                    target.eval(&args),
                    source.eval(&args),
                    "bv{width}: {x} + {y}"
                );
            }
        }
    }
    let (source, target) = split_add(4, true);
    for x in 0..16 {
        for y in 0..16 {
            let args = [Value::Bv(x), Value::Bv(y)];
            assert_eq!(target.eval(&args), source.eval(&args));
        }
    }
}

#[test]
fn missing_carry_has_a_counterexample() {
    let (source, target) = split_add(64, false);
    let args = [Value::Bv(u32::MAX as u128), Value::Bv(1)];
    assert_eq!(source.eval(&args), Ok(Value::Bv(1u128 << 32)));
    assert_eq!(target.eval(&args), Ok(Value::Bv(0)));
}

#[test]
fn query_uses_typed_signature_and_shared_subexpressions() {
    let (source, target) = split_add(64, true);
    let query = equivalence_query(&source, &target).unwrap();
    assert!(query.contains("(set-logic QF_BV)"));
    assert!(query.contains("(declare-fun x0 () (_ BitVec 64))"));
    assert!(query.contains("(declare-fun x1 () (_ BitVec 64))"));
    assert!(query.contains("((_ extract 31 0)"));
    assert!(query.contains("((_ zero_extend 31)"));
    assert!(query.contains("(bvult "));
    assert!(query.contains("(concat "));
    assert!(query.ends_with("(check-sat)\n"));
    // Source: one add. Target: low add, high add, carry add. The shared low
    // expression is emitted once, despite being used for both result and carry.
    assert_eq!(query.matches("(bvadd ").count(), 4);
    assert_eq!(query, equivalence_query(&source, &target).unwrap());
}

#[test]
fn typed_constructors_reject_invalid_terms() {
    let x = Expr::input(0, Sort::bv(32).unwrap());
    let y = Expr::input(1, Sort::bv(64).unwrap());
    assert!(matches!(
        Expr::apply(BvOp::Add, &[]),
        Err(Error::Arity { .. })
    ));
    assert!(matches!(
        Expr::apply(BvOp::Add, &[x.clone(), y.clone()]),
        Err(Error::SortMismatch { .. })
    ));
    assert!(matches!(
        Expr::apply(BvOp::Neg, &[Expr::bool(false)]),
        Err(Error::ExpectedBv(_))
    ));
    assert!(matches!(
        x.extract(32, 0),
        Err(Error::InvalidExtract { .. })
    ));
    assert!(matches!(x.extract(0, 1), Err(Error::InvalidExtract { .. })));
    assert!(matches!(
        x.zero_extend(16),
        Err(Error::InvalidExtension { .. })
    ));
    assert_eq!(x.zero_extend(0).unwrap_err(), Error::InvalidWidth(0));
    assert!(matches!(
        Expr::concat(&Expr::bv(128, 0).unwrap(), &x),
        Err(Error::InvalidWidth(160))
    ));
    assert!(matches!(x.ult(&y), Err(Error::SortMismatch { .. })));
    assert!(matches!(
        Expr::select(&x, &x, &x),
        Err(Error::SortMismatch { .. })
    ));
    assert!(matches!(
        Expr::select(&Expr::bool(true), &x, &y),
        Err(Error::SortMismatch { .. })
    ));
}

#[test]
fn function_rejects_undeclared_or_inconsistently_typed_inputs() {
    let bv32 = Sort::bv(32).unwrap();
    let bv64 = Sort::bv(64).unwrap();
    assert!(matches!(
        Function::new([], Expr::input(0, bv32)),
        Err(Error::InputIndex { .. })
    ));
    assert!(matches!(
        Function::new([bv32], Expr::input(0, bv64)),
        Err(Error::InputSort { .. })
    ));
    let x = Expr::input(0, bv32);
    let y = Expr::input(0, bv64).extract(31, 0).unwrap();
    assert!(matches!(
        Function::new([bv32], Expr::apply(BvOp::Add, &[x, y]).unwrap()),
        Err(Error::InputSort { .. })
    ));
}

#[test]
fn execution_and_queries_use_the_same_signature() {
    let bv8 = Sort::bv(8).unwrap();
    let f = Function::new([bv8, Sort::Bool], Expr::input(0, bv8)).unwrap();
    assert_eq!(
        f.eval(&[Value::Bv(511), Value::Bool(true)]),
        Ok(Value::Bv(255))
    );
    assert!(matches!(f.eval(&[Value::Bv(1)]), Err(Error::Arity { .. })));
    assert!(matches!(
        f.eval(&[Value::Bool(true), Value::Bool(false)]),
        Err(Error::ValueSort { .. })
    ));
    let g = Function::new([bv8], Expr::input(0, bv8)).unwrap();
    assert_eq!(equivalence_query(&f, &g), Err(Error::SignatureMismatch));
    let h = Function::new([bv8], Expr::bool(true)).unwrap();
    assert!(matches!(
        equivalence_query(&g, &h),
        Err(Error::SortMismatch { .. })
    ));
    let query = equivalence_query(&f, &f).unwrap();
    assert!(query.contains("(declare-fun x1 () Bool)"));
}

#[test]
fn shared_dag_and_boolean_results_execute_once_per_node() {
    let bv = Sort::bv(8).unwrap();
    let x = Expr::input(0, bv);
    let square = Expr::apply(BvOp::Mul, &[x.clone(), x.clone()]).unwrap();
    let sum = Expr::apply(BvOp::Add, &[square.clone(), square]).unwrap();
    let f = Function::new([bv], sum.ult(&x).unwrap()).unwrap();
    assert_eq!(f.eval(&[Value::Bv(16)]), Ok(Value::Bool(true)));
    let query = equivalence_query(&f, &f).unwrap();
    assert_eq!(query.matches("(bvmul ").count(), 2);
    let choice = Expr::select(&Expr::bool(false), &Expr::bool(true), &Expr::bool(false)).unwrap();
    let f = Function::new([], choice).unwrap();
    assert_eq!(f.eval(&[]), Ok(Value::Bool(false)));
}
