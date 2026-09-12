//! Instantiate recipes, check types, execute, and construct verification queries.
use veloc_semantics::{
    BvConst, BvOp, ComparisonRef, Conversion, Error, Expr, Function, IntPredicate, Program, Sort,
    Step, TypeRef, Value, Width, equivalence_query,
};

const NEG: Program = Program {
    inputs: 1,
    properties: 0,
    traps: &[],
    steps: &[
        Step::Input(0),
        Step::Apply {
            op: BvOp::Neg,
            args: &[0],
        },
    ],
    outputs: &[1],
};

const COMPOSED_NEG: Program = Program {
    inputs: 1,
    properties: 0,
    traps: &[],
    steps: &[
        Step::Const {
            value: BvConst::Zero,
            ty: veloc_semantics::TypeRef::Result(0),
        },
        Step::Input(0),
        Step::Apply {
            op: BvOp::Sub,
            args: &[0, 1],
        },
    ],
    outputs: &[2],
};

#[test]
fn composed_negation_agrees_with_the_primitive_at_every_width() {
    assert_eq!(NEG.primitive(), Some(BvOp::Neg));
    assert_eq!(COMPOSED_NEG.primitive(), None);
    for bits in 1..=128 {
        let width = Width::new(bits).unwrap();
        let function = COMPOSED_NEG.instantiate_width(width).unwrap();
        for x in [0, 1, width.mask() / 2, width.mask(), u128::MAX] {
            let result = NEG.eval(width, &[x]).unwrap();
            assert_eq!(COMPOSED_NEG.eval(width, &[x]), Ok(result));
            assert_eq!(function.eval(&[Value::Bv(x)]), Ok(Value::Bv(result)));
        }
    }
}

#[test]
fn primitive_recognition_uses_logical_input_order() {
    const SUB: Program = Program {
        inputs: 2,
        properties: 0,
        traps: &[],
        steps: &[
            Step::Const {
                value: BvConst::One,
                ty: veloc_semantics::TypeRef::Result(0),
            }, // Unused, but valid.
            Step::Input(1),
            Step::Input(0),
            Step::Apply {
                op: BvOp::Sub,
                args: &[2, 1],
            },
            Step::Const {
                value: BvConst::Zero,
                ty: veloc_semantics::TypeRef::Result(0),
            }, // The output need not be the last step.
        ],
        outputs: &[3],
    };
    assert_eq!(SUB.primitive(), Some(BvOp::Sub));
    assert_eq!(SUB.eval(Width::new(8).unwrap(), &[1, 2]), Ok(255));
    assert_eq!(Program { inputs: 3, ..SUB }.primitive(), None);
    assert_eq!(
        Program {
            outputs: &[0],
            ..SUB
        }
        .primitive(),
        None
    );
    const REVERSED: Program = Program {
        inputs: 2,
        properties: 0,
        traps: &[],
        steps: &[
            Step::Input(0),
            Step::Input(1),
            Step::Apply {
                op: BvOp::Add,
                args: &[1, 0],
            },
        ],
        outputs: &[2],
    };
    // Even commutativity does not turn recognition into proof search.
    assert_eq!(REVERSED.primitive(), None);
    const REPEATED: Program = Program {
        inputs: 2,
        properties: 0,
        traps: &[],
        steps: &[
            Step::Input(0),
            Step::Apply {
                op: BvOp::Add,
                args: &[0, 0],
            },
        ],
        outputs: &[1],
    };
    assert_eq!(REPEATED.primitive(), None);
}

#[test]
fn constant_and_identity_programs_preserve_their_signatures() {
    const CONSTANT: Program = Program {
        inputs: 0,
        properties: 0,
        traps: &[],
        steps: &[Step::Const {
            value: BvConst::AllOnes,
            ty: veloc_semantics::TypeRef::Result(0),
        }],
        outputs: &[0],
    };
    const IDENTITY: Program = Program {
        inputs: 2,
        properties: 0,
        traps: &[],
        steps: &[Step::Input(0)],
        outputs: &[0],
    };
    let width = Width::new(8).unwrap();
    assert_eq!(CONSTANT.eval(width, &[]), Ok(255));
    assert_eq!(
        CONSTANT.instantiate_width(width).unwrap().eval(&[]),
        Ok(Value::Bv(255))
    );
    assert_eq!(IDENTITY.eval(width, &[257, 0]), Ok(1));
    assert_eq!(
        IDENTITY.instantiate_width(width).unwrap().inputs(),
        &[Sort::Bv(width); 2]
    );
    assert_eq!(IDENTITY.primitive(), None);
    assert_eq!(
        IDENTITY.eval(width, &[1]),
        Err(Error::Arity {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn rejects_invalid_step_references_and_lengths_without_panicking() {
    const INVALID: &[Program] = &[
        Program {
            inputs: 0,
            properties: 0,
            traps: &[],
            steps: &[],
            outputs: &[0],
        },
        Program {
            inputs: 0,
            properties: 0,
            traps: &[],
            steps: &[Step::Input(0)],
            outputs: &[0],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[Step::Input(1)],
            outputs: &[0],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[Step::Input(0)],
            outputs: &[1],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[
                Step::Input(0),
                Step::Apply {
                    op: BvOp::Neg,
                    args: &[1],
                },
            ],
            outputs: &[1],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[
                Step::Input(0),
                Step::Apply {
                    op: BvOp::Neg,
                    args: &[2],
                },
                Step::Input(0),
            ],
            outputs: &[1],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[
                Step::Input(0),
                Step::Apply {
                    op: BvOp::Add,
                    args: &[0],
                },
            ],
            outputs: &[1],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[
                Step::Input(0),
                Step::Apply {
                    op: BvOp::Neg,
                    args: &[],
                },
            ],
            outputs: &[0],
        },
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[
                Step::Input(0),
                Step::Apply {
                    op: BvOp::Neg,
                    args: &[0, 0],
                },
            ],
            outputs: &[0],
        },
        // A direct primitive must not bypass validation of unrelated steps.
        Program {
            inputs: 1,
            properties: 0,
            traps: &[],
            steps: &[
                Step::Input(0),
                Step::Apply {
                    op: BvOp::Neg,
                    args: &[0],
                },
                Step::Input(1),
            ],
            outputs: &[1],
        },
    ];
    let width = Width::new(32).unwrap();
    for program in INVALID {
        assert!(program.validate().is_err(), "{program:?}");
        assert!(program.eval(width, &[0]).is_err(), "{program:?}");
        assert!(program.instantiate_width(width).is_err(), "{program:?}");
        assert_eq!(program.primitive(), None);
    }
}

#[test]
fn step_count_must_fit_the_reference_representation() {
    const MAX: usize = u16::MAX as usize + 1;
    static TOO_LONG: [Step; MAX + 1] = [Step::Const {
        value: BvConst::Zero,
        ty: veloc_semantics::TypeRef::Result(0),
    }; MAX + 1];
    static FULL: [Step; MAX] = [Step::Const {
        value: BvConst::One,
        ty: veloc_semantics::TypeRef::Result(0),
    }; MAX];
    let program = Program {
        inputs: 0,
        properties: 0,
        traps: &[],
        steps: &TOO_LONG,
        outputs: &[0],
    };
    assert_eq!(
        program.validate(),
        Err(Error::TooManySteps {
            count: MAX + 1,
            max: MAX
        })
    );
    assert!(program.instantiate_width(Width::new(8).unwrap()).is_err());
    assert!(program.eval(Width::new(8).unwrap(), &[]).is_err());
    assert_eq!(program.primitive(), None);
    let full = Program {
        inputs: 0,
        properties: 0,
        traps: &[],
        steps: &FULL,
        outputs: &[u16::MAX],
    };
    assert_eq!(full.validate(), Ok(()));
    assert_eq!(full.eval(Width::new(8).unwrap(), &[]), Ok(1));
}

#[test]
fn composed_programs_share_the_existing_smt_encoder() {
    let width = Width::new(64).unwrap();
    let query = equivalence_query(
        &NEG.instantiate_width(width).unwrap(),
        &COMPOSED_NEG.instantiate_width(width).unwrap(),
    )
    .unwrap();
    assert!(query.contains("(declare-fun x0 () (_ BitVec 64))"));
    assert!(!query.contains("(declare-fun x1"));
    assert!(query.contains("(bvneg "));
    assert!(query.contains("(bvsub "));
    assert!(query.contains("(_ bv0 64)"));
    assert!(query.ends_with("(check-sat)\n"));
}

trait ScalarProgram {
    fn instantiate_width(&self, width: Width) -> Result<veloc_semantics::Function, Error>;
    fn eval(&self, width: Width, args: &[u128]) -> Result<u128, Error>;
}
impl ScalarProgram for Program<'_> {
    fn instantiate_width(&self, width: Width) -> Result<veloc_semantics::Function, Error> {
        self.instantiate(
            &vec![Sort::Bv(width); self.inputs as usize],
            &vec![Sort::Bv(width); self.outputs.len()],
            &[],
        )
    }
    fn eval(&self, width: Width, args: &[u128]) -> Result<u128, Error> {
        let values = args.iter().copied().map(Value::Bv).collect::<Vec<_>>();
        match self.instantiate_width(width)?.eval(&values)? {
            Value::Bv(v) => Ok(v),
            _ => panic!("test expected bitvector"),
        }
    }
}

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
