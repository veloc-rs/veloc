use veloc_semantics::{BvConst, BvOp};

use crate::Error;
use crate::model::{Op, Param, ParamKind, Pattern, Semantic, SemanticStep};
use crate::syntax::{Kind, Node};

/// Compile an expression into width-parameterized semantic steps. SSA input
/// numbering follows logical value parameters, independent of physical packing.
pub(crate) fn parse(source: &str, node: Node, params: &[Param]) -> Result<Semantic, Error> {
    let offset = node.offset;
    if params.iter().any(|param| {
        matches!(
            param.kind,
            ParamKind::Values | ParamKind::Successor | ParamKind::Successors
        )
    }) {
        return Err(Error::at(
            source,
            offset,
            "pure bitvector semantics cannot use variadic values or successors",
        ));
    }
    let inputs = u8::try_from(
        params
            .iter()
            .filter(|param| param.kind == ParamKind::Value)
            .count(),
    )
    .map_err(|_| Error::at(source, offset, "semantic input count exceeds 255"))?;
    let mut steps = (0..inputs).map(SemanticStep::Input).collect::<Vec<_>>();
    let output = expression(source, node, params, &mut steps)?;
    Ok(Semantic {
        steps,
        output,
        inputs,
    })
}

fn expression(
    source: &str,
    node: Node,
    params: &[Param],
    steps: &mut Vec<SemanticStep>,
) -> Result<u16, Error> {
    let offset = node.offset;
    match node.kind {
        Kind::Name(name) => {
            let mut index = 0;
            for param in params {
                if param.name == name {
                    return if param.kind == ParamKind::Value {
                        Ok(index)
                    } else {
                        Err(Error::at(
                            source,
                            offset,
                            format!("semantic expression cannot reference property `{name}`"),
                        ))
                    };
                }
                if param.kind == ParamKind::Value {
                    index += 1;
                }
            }
            Err(Error::at(
                source,
                offset,
                format!("unknown semantic input `{name}`"),
            ))
        }
        Kind::Call(name, args) => {
            let constant = match name.as_str() {
                "bv.zero" => Some(BvConst::Zero),
                "bv.one" => Some(BvConst::One),
                "bv.ones" => Some(BvConst::AllOnes),
                _ => None,
            };
            if let Some(constant) = constant {
                if !args.is_empty() {
                    return Err(Error::at(
                        source,
                        offset,
                        format!("semantic constant `{name}` expects no arguments"),
                    ));
                }
                return push(source, offset, steps, SemanticStep::Const(constant));
            }
            let op = BvOp::from_name(&name).ok_or_else(|| {
                Error::at(
                    source,
                    offset,
                    format!("unknown semantic primitive `{name}`"),
                )
            })?;
            if args.len() != op.arity() {
                return Err(Error::at(
                    source,
                    offset,
                    format!(
                        "{} expects {} arguments, got {}",
                        op.name(),
                        op.arity(),
                        args.len()
                    ),
                ));
            }
            let args = args
                .into_iter()
                .map(|arg| expression(source, arg, params, steps))
                .collect::<Result<Vec<_>, _>>()?;
            push(source, offset, steps, SemanticStep::Apply { op, args })
        }
        _ => Err(Error::at(
            source,
            offset,
            "expected a named value or bitvector semantic call",
        )),
    }
}

fn push(
    source: &str,
    offset: usize,
    steps: &mut Vec<SemanticStep>,
    step: SemanticStep,
) -> Result<u16, Error> {
    let index = u16::try_from(steps.len())
        .map_err(|_| Error::at(source, offset, "semantic program exceeds 65536 steps"))?;
    steps.push(step);
    Ok(index)
}

pub(crate) fn validate(
    source: &str,
    op: &Op,
    types: &crate::types::Types,
    builtins: &crate::builtins::Builtins,
) -> Result<(), Error> {
    let Some(semantics) = &op.semantics else {
        return Ok(());
    };
    let ty = &op.signature;
    let fail = || {
        Error::at(
            source,
            op.offset,
            "bitvector semantics require pure, same-width integer inputs and one integer result",
        )
    };
    if !builtins.effects[&op.memory].is_none()
        || op
            .traits
            .iter()
            .any(|name| matches!(name.as_str(), "MAY_TRAP" | "TERMINATOR"))
        || !ty.relations.is_empty()
    {
        return Err(fail());
    }
    let (Some(args), Some([result])) = (ty.operands.patterns(), ty.results.patterns()) else {
        return Err(fail());
    };
    if args.len() != usize::from(semantics.inputs) {
        return Err(fail());
    }
    let integer_class = |set: &crate::type_set::TypeSet| set.subset_of(&types.integers);
    let integer_type = |name: &str| types.exact[name].subset_of(&types.integers);
    let compatible = match args.first() {
        Some(Pattern::Bind(var, class)) if integer_class(class) => {
            args[1..].iter().chain([result]).all(|pattern| {
                matches!(pattern, Pattern::Same(other) | Pattern::Bind(other, _) if var == other)
            })
        }
        Some(Pattern::Exact(name)) if integer_type(name) => {
            args[1..].iter().chain([result]).all(|pattern| {
                matches!(pattern, Pattern::Exact(other) if other == name)
            })
        }
        None => matches!(result,
            Pattern::Class(class) | Pattern::Bind(_, class) if integer_class(class)
        ) || matches!(result, Pattern::Exact(name) if integer_type(name)),
        _ => false,
    };
    if compatible { Ok(()) } else { Err(fail()) }
}

/// Only direct primitive applications inherit reviewed algebraic facts. A
/// composed expression needs a separate proof before algebraic rewrites can use
/// it; structural resemblance alone does not establish those facts.
pub(crate) fn derive(
    source: &str,
    offset: usize,
    semantics: &Semantic,
    traits: &mut Vec<String>,
    identity: &mut Option<BvConst>,
    absorbing: &mut Option<BvConst>,
) -> Result<(), Error> {
    const ALGEBRAIC: [&str; 3] = ["COMMUTATIVE", "ASSOCIATIVE", "IDEMPOTENT"];
    let Some(primitive) = semantics.primitive() else {
        if identity.is_some()
            || absorbing.is_some()
            || traits.iter().any(|name| ALGEBRAIC.contains(&name.as_str()))
        {
            return Err(Error::at(
                source,
                offset,
                "composed semantics require a proof before declaring algebraic properties",
            ));
        }
        return Ok(());
    };
    let facts = primitive.algebra();
    for (name, supported) in [
        ("COMMUTATIVE", facts.commutative),
        ("ASSOCIATIVE", facts.associative),
        ("IDEMPOTENT", facts.idempotent),
    ] {
        let declared = traits.iter().any(|value| value == name);
        if declared && !supported {
            return Err(Error::at(
                source,
                offset,
                format!(
                    "{} does not support trait `{name}` at every supported width",
                    primitive.name()
                ),
            ));
        }
        if supported && !declared {
            traits.push(name.into());
        }
    }
    for (name, declared, expected) in [
        ("identity", identity, facts.identity),
        ("absorbing", absorbing, facts.absorbing),
    ] {
        if let Some(value) = *declared
            && expected != Some(value)
        {
            return Err(Error::at(
                source,
                offset,
                format!(
                    "{} does not support {name} `{value:?}` at every supported width",
                    primitive.name()
                ),
            ));
        }
        *declared = expected;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TypeDef, TypeList};

    fn params() -> Vec<Param> {
        vec![
            Param {
                name: "lhs".into(),
                kind: ParamKind::Value,
            },
            Param {
                name: "flag".into(),
                kind: ParamKind::Property("bool".into()),
            },
            Param {
                name: "rhs".into(),
                kind: ParamKind::Value,
            },
        ]
    }

    fn parsed(expression: &str, params: &[Param]) -> Result<Semantic, Error> {
        let source = format!("format Test {{ semantics: {expression} }}");
        let mut record = crate::syntax::parse(&source)?.pop().unwrap();
        parse(&source, record.fields.remove("semantics").unwrap(), params)
    }

    #[test]
    fn named_nested_calls_use_logical_value_input_order() {
        let sem = parsed("bv.add(lhs, bv.sub(rhs, bv.one()))", &params()).unwrap();
        assert_eq!(sem.inputs, 2);
        assert_eq!(sem.output, 4);
        assert!(matches!(sem.steps[0], SemanticStep::Input(0)));
        assert!(matches!(sem.steps[1], SemanticStep::Input(1)));
        assert!(matches!(sem.steps[2], SemanticStep::Const(BvConst::One)));
        assert!(
            matches!(&sem.steps[3], SemanticStep::Apply { op: BvOp::Sub, args } if args == &[1, 2])
        );
        assert!(
            matches!(&sem.steps[4], SemanticStep::Apply { op: BvOp::Add, args } if args == &[0, 3])
        );
        assert_eq!(sem.primitive(), None);
    }

    #[test]
    fn semantic_names_and_arities_are_checked() {
        for expression in [
            "missing",
            "flag",
            "bv.unknown(lhs)",
            "bv.add(lhs)",
            "bv.neg(lhs, rhs)",
            "bv.zero(lhs)",
            "[lhs]",
            "42",
        ] {
            assert!(parsed(expression, &params()).is_err(), "{expression}");
        }
        for (expression, constant) in [
            ("bv.zero()", BvConst::Zero),
            ("bv.one()", BvConst::One),
            ("bv.ones()", BvConst::AllOnes),
        ] {
            let sem = parsed(expression, &[]).unwrap();
            assert!(matches!(sem.steps[0], SemanticStep::Const(value) if value == constant));
        }
    }

    #[test]
    fn semantic_programs_reject_contextual_parameters_and_index_overflow() {
        for kind in [
            ParamKind::Values,
            ParamKind::Successor,
            ParamKind::Successors,
        ] {
            assert!(
                parsed(
                    "bv.zero()",
                    &[Param {
                        name: "args".into(),
                        kind
                    }]
                )
                .is_err()
            );
        }
        let params = (0..256)
            .map(|i| Param {
                name: format!("v{i}"),
                kind: ParamKind::Value,
            })
            .collect::<Vec<_>>();
        assert!(parsed("v0", &params).is_err());
        let mut steps = (0..65535)
            .map(|_| SemanticStep::Const(BvConst::Zero))
            .collect::<Vec<_>>();
        assert_eq!(
            push("", 0, &mut steps, SemanticStep::Const(BvConst::One)),
            Ok(u16::MAX)
        );
        assert!(push("", 0, &mut steps, SemanticStep::Const(BvConst::One)).is_err());
    }

    #[test]
    fn direct_primitives_supply_facts_without_repeated_declarations() {
        let sem = parsed("bv.and(lhs, rhs)", &params()).unwrap();
        let (mut traits, mut identity, mut absorbing) = (vec![], None, None);
        derive("", 0, &sem, &mut traits, &mut identity, &mut absorbing).unwrap();
        assert_eq!(traits, ["COMMUTATIVE", "ASSOCIATIVE", "IDEMPOTENT"]);
        assert_eq!(identity, Some(BvConst::AllOnes));
        assert_eq!(absorbing, Some(BvConst::Zero));
        derive("", 0, &sem, &mut traits, &mut identity, &mut absorbing).unwrap();
        assert_eq!(traits.len(), 3);
    }

    #[test]
    fn contradictory_and_unproved_facts_are_rejected() {
        let mut identity = Some(BvConst::One);
        assert!(
            derive(
                "",
                0,
                &parsed("bv.add(lhs, rhs)", &params()).unwrap(),
                &mut vec![],
                &mut identity,
                &mut None
            )
            .is_err()
        );
        assert!(
            derive(
                "",
                0,
                &parsed("bv.sub(lhs, rhs)", &params()).unwrap(),
                &mut vec!["COMMUTATIVE".into()],
                &mut None,
                &mut None
            )
            .is_err()
        );
        let composed = parsed(
            "bv.sub(bv.zero(), lhs)",
            &[Param {
                name: "lhs".into(),
                kind: ParamKind::Value,
            }],
        )
        .unwrap();
        assert!(derive("", 0, &composed, &mut vec![], &mut None, &mut None).is_ok());
        assert!(
            derive(
                "",
                0,
                &composed,
                &mut vec!["COMMUTATIVE".into()],
                &mut None,
                &mut None
            )
            .is_err()
        );
    }

    fn unary(operand: Pattern, result: Pattern) -> Op {
        let params = vec![Param {
            name: "arg".into(),
            kind: ParamKind::Value,
        }];
        let sem = parsed("bv.neg(arg)", &params).unwrap();
        Op {
            offset: 0,
            name: "Test".into(),
            mnemonic: "test".into(),
            format: "Unary".into(),
            signature: TypeDef {
                operands: TypeList::Fixed(vec![operand]),
                results: TypeList::Fixed(vec![result]),
                relations: vec![],
            },
            signature_source: None,
            text: None,
            params,
            packing: Default::default(),
            traits: vec![],
            memory: "NONE".into(),
            constraints: vec![],
            identity: None,
            absorbing: None,
            semantics: Some(sem),
        }
    }

    #[test]
    fn same_width_integer_schemes_include_vectors_and_exact_types() {
        for class in [
            "Integer",
            "ScalarInteger",
            "Integer | BOOL | vectors(BOOL)",
            "Integer & Vector",
        ] {
            let op = unary(
                Pattern::Bind(0, crate::fixtures::set(class)),
                Pattern::Same(0),
            );
            validate(
                "",
                &op,
                &crate::fixtures::types(),
                &crate::fixtures::builtins(),
            )
            .unwrap();
        }
        for name in ["I8", "I16", "I32", "I64", "BOOL"] {
            let op = unary(Pattern::Exact(name.into()), Pattern::Exact(name.into()));
            validate(
                "",
                &op,
                &crate::fixtures::types(),
                &crate::fixtures::builtins(),
            )
            .unwrap();
        }
        for (operand, result) in [
            (Pattern::Exact("I32".into()), Pattern::Exact("I64".into())),
            (
                Pattern::Bind(0, crate::fixtures::set("Float")),
                Pattern::Same(0),
            ),
            (
                Pattern::Class(crate::fixtures::set("ScalarInteger")),
                Pattern::Class(crate::fixtures::set("ScalarInteger")),
            ),
        ] {
            let op = unary(operand, result);
            assert!(
                validate(
                    "",
                    &op,
                    &crate::fixtures::types(),
                    &crate::fixtures::builtins()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn executable_bitvector_semantics_do_not_claim_effects_or_traps() {
        let mut op = unary(Pattern::Exact("I32".into()), Pattern::Exact("I32".into()));
        op.memory = "UNKNOWN".into();
        assert!(
            validate(
                "",
                &op,
                &crate::fixtures::types(),
                &crate::fixtures::builtins()
            )
            .is_err()
        );
        op.memory = "NONE".into();
        for flag in ["MAY_TRAP", "TERMINATOR"] {
            op.traits = vec![flag.into()];
            assert!(
                validate(
                    "",
                    &op,
                    &crate::fixtures::types(),
                    &crate::fixtures::builtins()
                )
                .is_err()
            );
        }
        op.traits.clear();
        op.signature.results = TypeList::Signature;
        assert!(
            validate(
                "",
                &op,
                &crate::fixtures::types(),
                &crate::fixtures::builtins()
            )
            .is_err()
        );
        op.semantics = None;
        validate(
            "",
            &op,
            &crate::fixtures::types(),
            &crate::fixtures::builtins(),
        )
        .unwrap();
    }
}
