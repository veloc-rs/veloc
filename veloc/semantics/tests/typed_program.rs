use veloc_semantics::{
    ComparisonRef, Conversion, Error, Expr, Function, IntPredicate, Program, Sort, Step, TypeRef,
    Value, equivalence_query,
};

#[test]
fn output_order_and_sharing_are_independent_of_graph_order() {
    let sort = Sort::bv(8).unwrap();
    let x = Expr::input(0, sort);
    let wide = x.zero_extend(16).unwrap();
    let function = Function::with_outputs([sort], [wide.clone(), x.clone(), wide]).unwrap();
    assert_eq!(
        function.eval_all(&[Value::Bv(511)]).unwrap(),
        [Value::Bv(255); 3]
    );
    assert!(matches!(
        function.eval(&[Value::Bv(0)]),
        Err(Error::ResultArity { .. })
    ));
    let other = Function::with_outputs([sort], [x.clone(), x.clone()]).unwrap();
    assert!(matches!(
        equivalence_query(&function, &other),
        Err(Error::ResultArity { .. })
    ));
    let source = Function::with_outputs([sort], [x.clone(), Expr::bool(false)]).unwrap();
    let target = Function::with_outputs([sort], [x, Expr::bool(true)]).unwrap();
    let query = equivalence_query(&source, &target).unwrap();
    assert!(query.contains("(assert (or (not (= s0 t0)) (not (= s1 t1))))"));
}

#[test]
fn signature_properties_and_result_types_are_checked() {
    let program: Program = Program {
        inputs: 2,
        properties: 1,
        traps: &[],
        steps: &[
            Step::Input(0),
            Step::Input(1),
            Step::Compare {
                kind: ComparisonRef::Property(0),
                lhs: 0,
                rhs: 1,
            },
        ],
        outputs: &[2],
    };
    let bv8 = Sort::bv(8).unwrap();
    let pred = IntPredicate::new(false, 1);
    assert!(
        program
            .instantiate(&[bv8, bv8], &[Sort::Bool], &[pred])
            .is_ok()
    );
    assert!(
        program
            .instantiate(&[bv8, bv8], &[Sort::Bool], &[])
            .is_err()
    );
    assert!(
        program
            .instantiate(&[bv8, Sort::bv(16).unwrap()], &[Sort::Bool], &[pred])
            .is_err()
    );
    assert!(program.instantiate(&[bv8, bv8], &[bv8], &[pred]).is_err());
    assert!(
        program
            .instantiate(&[Sort::Bool, Sort::Bool], &[Sort::Bool], &[pred])
            .is_err()
    );
    assert!(
        Program {
            properties: 0,
            ..program
        }
        .validate()
        .is_err()
    );
    assert!(
        Program {
            outputs: &[3],
            ..program
        }
        .validate()
        .is_err()
    );
}

#[test]
fn even_unused_steps_receive_shared_expression_type_checks() {
    let bv8 = Sort::bv(8).unwrap();
    let program: Program = Program {
        inputs: 1,
        properties: 0,
        traps: &[],
        steps: &[
            Step::Input(0),
            Step::Convert {
                kind: Conversion::SignExtend,
                arg: 0,
                to: TypeRef::Result(1),
            },
        ],
        outputs: &[0, 0],
    };
    assert!(program.instantiate(&[bv8], &[bv8, bv8], &[]).is_ok());
    assert!(
        program
            .instantiate(&[bv8], &[bv8, Sort::Bool], &[])
            .is_err()
    );
    assert!(
        Program {
            outputs: &[0],
            ..program
        }
        .validate()
        .is_err()
    );
}
