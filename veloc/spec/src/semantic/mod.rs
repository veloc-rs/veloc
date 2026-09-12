use crate::Error;
pub(crate) mod emit;
use crate::model::{Op, Param, ParamKind, Pattern, Semantic, SemanticStep};
use crate::syntax::{Kind, Node};
use veloc_semantics::{
    BvConst, BvOp, ComparisonRef, Conversion, IntPredicate, Sort, Trap, TypeRef,
};

pub(crate) fn parse(source: &str, node: Node, params: &[Param]) -> Result<Semantic, Error> {
    if params.iter().any(|p| {
        matches!(
            p.kind,
            ParamKind::Values | ParamKind::Successor | ParamKind::Successors
        )
    }) {
        return Err(Error::at(
            source,
            node.offset,
            "pure semantics cannot use variadic values or successors",
        ));
    }
    let inputs = u8::try_from(params.iter().filter(|p| p.kind == ParamKind::Value).count())
        .map_err(|_| Error::at(source, node.offset, "semantic input count exceeds 255"))?;
    let properties = params
        .iter()
        .filter(|p| matches!(&p.kind, ParamKind::Property(ty) if ty == "IntCC"))
        .map(|p| p.name.clone())
        .collect::<Vec<_>>();
    if properties.len() > u8::MAX as usize {
        return Err(Error::at(
            source,
            node.offset,
            "semantic property count exceeds 255",
        ));
    }
    let mut sem = Semantic {
        inputs,
        properties,
        steps: (0..inputs).map(SemanticStep::Input).collect(),
        outputs: Vec::new(),
        traps: Vec::new(),
        instances: Vec::new(),
    };
    let outputs = match node.kind {
        Kind::List(outputs) => outputs,
        _ => vec![node],
    };
    if outputs.is_empty() || outputs.len() > u8::MAX as usize {
        return Err(Error::at(source, 0, "semantics require 1..=255 results"));
    }
    for output in outputs {
        let index = expression(source, output, params, &mut sem)?;
        sem.outputs.push(index);
    }
    Ok(sem)
}

pub(crate) fn traps(
    source: &str,
    node: Node,
    params: &[Param],
    sem: &mut Semantic,
) -> Result<(), Error> {
    for node in crate::model::list(source, node)? {
        let Kind::Call(name, mut args) = node.kind else {
            return Err(Error::at(
                source,
                node.offset,
                "expected TrapName(condition)",
            ));
        };
        let trap = match name.as_str() {
            "DivisionByZero" => Trap::DivisionByZero,
            "IntegerOverflow" => Trap::IntegerOverflow,
            _ => {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown trap `{name}`"),
                ));
            }
        };
        if args.len() != 1 {
            return Err(Error::at(
                source,
                node.offset,
                "trap requires one boolean condition",
            ));
        }
        let guard = expression(source, args.remove(0), params, sem)?;
        sem.traps.push((guard, trap));
    }
    Ok(())
}

fn type_ref(source: &str, node: Node, params: &[Param]) -> Result<TypeRef, Error> {
    match node.kind {
        Kind::Call(name, args) if name == "result" && args.len() == 1 => {
            if let Kind::Number(index) = args[0].kind
                && let Ok(index) = u8::try_from(index)
            {
                return Ok(TypeRef::Result(index));
            }
        }
        Kind::Call(name, args) if name == "type" && args.len() == 1 => {
            let name = crate::model::name(source, args[0].clone())?;
            if let Some(index) = params
                .iter()
                .filter(|p| p.kind == ParamKind::Value)
                .position(|p| p.name == name)
            {
                return Ok(TypeRef::Input(index as u8));
            }
        }
        _ => {}
    }
    Err(Error::at(
        source,
        node.offset,
        "expected type(operand) or result(index)",
    ))
}

fn expression(
    source: &str,
    node: Node,
    params: &[Param],
    sem: &mut Semantic,
) -> Result<u16, Error> {
    let offset = node.offset;
    let fail = |message| Error::at(source, offset, message);
    // Primitive namespaces use the same structural postfix syntax as methods.
    let kind = match node.kind {
        Kind::Method(receiver, name, args) => {
            let Kind::Name(namespace) = receiver.kind else {
                return Err(fail("semantic primitive requires a namespace".into()));
            };
            Kind::Call(format!("{namespace}.{name}"), args)
        }
        kind => kind,
    };
    let step = match kind {
        Kind::Name(name) => {
            return params
                .iter()
                .filter(|p| p.kind == ParamKind::Value)
                .position(|p| p.name == name)
                .map(|i| i as u16)
                .ok_or_else(|| fail(format!("unknown semantic value `{name}`")));
        }
        Kind::Call(name, mut args) => {
            if let Some(value) = match name.as_str() {
                "bv.zero" => Some(BvConst::Zero),
                "bv.one" => Some(BvConst::One),
                "bv.ones" => Some(BvConst::AllOnes),
                "bv.width" => Some(BvConst::Width),
                "bv.smin" => Some(BvConst::SignedMin),
                _ => None,
            } {
                let ty = match args.len() {
                    0 => {
                        if sem.inputs == 0 {
                            TypeRef::Result(0)
                        } else {
                            TypeRef::Input(0)
                        }
                    }
                    1 => type_ref(source, args.remove(0), params)?,
                    _ => return Err(fail("constant expects at most one type reference".into())),
                };
                SemanticStep::Const { value, ty }
            } else if matches!(name.as_str(), "bv.zext" | "bv.sext" | "bv.trunc") {
                if args.len() != 2 {
                    return Err(fail("conversion expects a value and result type".into()));
                }
                let to = type_ref(source, args.pop().unwrap(), params)?;
                let arg = expression(source, args.pop().unwrap(), params, sem)?;
                let kind = match name.as_str() {
                    "bv.zext" => Conversion::ZeroExtend,
                    "bv.sext" => Conversion::SignExtend,
                    _ => Conversion::Truncate,
                };
                SemanticStep::Convert { kind, arg, to }
            } else if matches!(name.as_str(), "bv.cmp" | "bv.slt" | "bv.ult" | "bv.eq") {
                let kind = if name == "bv.cmp" {
                    if args.len() != 3 {
                        return Err(fail(
                            "bv.cmp expects a comparison property and two operands".into(),
                        ));
                    }
                    let property = crate::model::name(source, args.remove(0))?;
                    let index = sem
                        .properties
                        .iter()
                        .position(|p| p == &property)
                        .ok_or_else(|| {
                            fail(format!("unknown integer comparison property `{property}`"))
                        })?;
                    ComparisonRef::Property(index as u8)
                } else {
                    ComparisonRef::Fixed(IntPredicate::new(
                        name == "bv.slt",
                        if name == "bv.eq" { 2 } else { 1 },
                    ))
                };
                if args.len() != 2 {
                    return Err(fail("comparison expects two operands".into()));
                }
                let lhs = expression(source, args.remove(0), params, sem)?;
                let rhs = expression(source, args.remove(0), params, sem)?;
                SemanticStep::Compare { kind, lhs, rhs }
            } else if name == "select" {
                if args.len() != 3 {
                    return Err(fail("select expects three operands".into()));
                }
                let cond = expression(source, args.remove(0), params, sem)?;
                let yes = expression(source, args.remove(0), params, sem)?;
                let no = expression(source, args.remove(0), params, sem)?;
                SemanticStep::Select { cond, yes, no }
            } else {
                let op = BvOp::from_name(&name)
                    .ok_or_else(|| fail(format!("unknown semantic primitive `{name}`")))?;
                if args.len() != op.arity() {
                    return Err(fail(format!(
                        "{} expects {} arguments, got {}",
                        op.name(),
                        op.arity(),
                        args.len()
                    )));
                }
                let args = args
                    .into_iter()
                    .map(|arg| expression(source, arg, params, sem))
                    .collect::<Result<_, _>>()?;
                SemanticStep::Apply { op, args }
            }
        }
        _ => return Err(fail("expected a named value or semantic call".into())),
    };
    // Preserve sharing in expressions such as a result also used to compute carry.
    if let Some(index) = sem.steps.iter().position(|existing| *existing == step) {
        return Ok(index as u16);
    }
    let index = u16::try_from(sem.steps.len())
        .map_err(|_| fail("semantic program exceeds 65536 steps".into()))?;
    sem.steps.push(step);
    Ok(index)
}

/// Validate every scalar element type admitted by the signature. The compact
/// type universe is finite; no solver or sampled-width proof is involved.
/// Shape-changing operations are not admitted as per-lane semantic recipes.
pub(crate) fn validate(
    source: &str,
    op: &Op,
    types: &crate::types::Types,
) -> Result<Vec<Instance>, Error> {
    let Some(sem) = &op.semantics else {
        return Ok(Vec::new());
    };
    let fail = |message| Error::at(source, op.offset, message);
    if !op.memory.is_none() || op.traits.iter().any(|t| t == "TERMINATOR") {
        return Err(fail(
            "executable semantics require no memory effects or control flow".into(),
        ));
    }
    if op.traits.iter().any(|t| t == "MAY_TRAP") != !sem.traps.is_empty() {
        return Err(fail(
            "MAY_TRAP must agree with explicit semantic trap guards".into(),
        ));
    }
    let (Some(inputs), Some(outputs)) = (
        op.signature.operands.patterns(),
        op.signature.results.patterns(),
    ) else {
        return Err(fail(
            "semantics require fixed input and result signatures".into(),
        ));
    };
    if outputs.len() != sem.outputs.len() {
        return Err(fail(
            "semantic result count does not match signature".into(),
        ));
    }
    sem.program().validate().map_err(|e| fail(e.to_string()))?;
    let instances = instances(source, op, types)?;
    for instance in &instances {
        let (ins, outs) = instance.sorts.split_at(inputs.len());
        let properties = vec![IntPredicate::new(false, 2); sem.properties.len()];
        sem.program()
            .instantiate(ins, outs, &properties)
            .map_err(|e| fail(format!("invalid semantic types {ins:?} -> {outs:?}: {e}")))?;
    }
    Ok(instances)
}

/// Concrete lane signatures shared by semantic validation and code generation.
/// `scalar` distinguishes genuinely scalar signatures from vector-only recipes.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Instance {
    pub kinds: Vec<crate::types::Primitive>,
    pub sorts: Vec<Sort>,
    pub scalar: bool,
}

fn instances(source: &str, op: &Op, types: &crate::types::Types) -> Result<Vec<Instance>, Error> {
    let constraints: Vec<_> = op.constraints.iter().filter(|c| c.type_only).collect();
    for constraint in &constraints {
        if !constraint.offline {
            return Err(Error::at(
                source,
                op.offset,
                "type constraint uses a Rust binding without offline evaluation support",
            ));
        }
    }
    let sem = op.semantics.as_ref().expect("semantic operation");
    let inputs = op.signature.operands.patterns().expect("fixed signature");
    let outputs = op.signature.results.patterns().expect("fixed signature");
    let fail = |message: String| Error::at(source, op.offset, message);
    // Pointer values are only modeled for comparisons. Their target width is
    // bound externally, never silently inferred from the build host.
    let mut instances = Vec::new();
    // Enumerate each independent type variable, not an assumed common shape.
    // Ordinary checked constraints determine which combinations are admissible.
    let mut bindings = std::collections::BTreeMap::new();
    let mut domains = Vec::new();
    for pattern in inputs.iter().chain(outputs) {
        let domain = match pattern {
            Pattern::Set(set) => Domain::Set(set),
            Pattern::Bind(var, set) => {
                bindings.insert(*var, domains.len());
                Domain::Set(set)
            }
            Pattern::Same(var) => Domain::Same(bindings[var]),
            Pattern::Exact(name) => Domain::Set(&types.exact[name]),
            _ => {
                return Err(fail(
                    "shape-changing semantic recipes are not supported".into(),
                ));
            }
        };
        domains.push(domain);
    }
    let mut accepted = std::collections::BTreeMap::<Vec<crate::types::Primitive>, bool>::new();
    enumerate(&domains, &mut Vec::new(), &mut 0, &mut |values| {
        if !constraints.iter().all(|c| {
            c.condition
                .accepts_types(types, &op.params, values, inputs.len())
        }) {
            return Ok(());
        }
        // A recipe describes one lane. Broadcasts, reductions and permutations
        // need their own model, even if their widths happen to agree.
        if values.windows(2).any(|pair| pair[0].1 != pair[1].1) {
            return Err("semantic recipes require a shared lane shape");
        }
        let scalar = values.iter().all(|(_, shape)| *shape == 0);
        *accepted
            .entry(values.iter().map(|(code, _)| *code).collect())
            .or_default() |= scalar;
        Ok(())
    })
    .map_err(|message| fail(message.into()))?;
    for (kinds, scalar) in accepted {
        let has_pointer = kinds.contains(&crate::types::Primitive::Ptr);
        let widths: &[u16] = if has_pointer { &[32, 64] } else { &[32] };
        for &pointer_width in widths {
            let sorts = kinds
                .iter()
                .map(|&kind| {
                    match kind {
                        crate::types::Primitive::Int(bits) => {
                            Ok(Sort::bv(bits as u16).unwrap())
                        }
                        crate::types::Primitive::Bool => Ok(Sort::Bool),
                        crate::types::Primitive::Ptr => {
                            let comparison_only = sem.steps.iter().all(|step| matches!(
                                step,
                                SemanticStep::Input(_) | SemanticStep::Compare {
                                    kind: ComparisonRef::Property(_), ..
                                }
                            ));
                            if !comparison_only {
                                return Err(fail(
                                    "pointer semantics only support comparison properties, not pointer arithmetic".into()
                                ));
                            }
                            Ok(Sort::bv(pointer_width).unwrap())
                        }
                        crate::types::Primitive::Float(_) => Err(fail(
                            "floating-point execution semantics are not modeled".into()
                        )),
                    }
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let instance = Instance {
                kinds: kinds.clone(),
                sorts,
                scalar,
            };
            instances.push(instance);
        }
    }
    if instances.is_empty() {
        return Err(fail("no admissible semantic signature".into()));
    }
    Ok(instances)
}

/// Signature domains preserve exact generic equality while allowing arbitrary
/// declared relationships between independent variables to be checked normally.
enum Domain<'a> {
    Set(&'a crate::types::TypeSet),
    Same(usize),
}

fn enumerate(
    domains: &[Domain<'_>],
    values: &mut Vec<(crate::types::Primitive, u32)>,
    visits: &mut usize,
    accept: &mut impl FnMut(&[(crate::types::Primitive, u32)]) -> Result<(), &'static str>,
) -> Result<(), &'static str> {
    *visits += 1;
    if *visits > 1_000_000 {
        return Err("semantic signature has too many type combinations");
    }
    let Some(domain) = domains.get(values.len()) else {
        return accept(values);
    };
    match domain {
        Domain::Same(index) => {
            values.push(values[*index]);
            enumerate(domains, values, visits, accept)?;
            values.pop();
        }
        Domain::Set(set) => {
            for (&code, &mask) in &set.0 {
                let mut shapes = mask;
                while shapes != 0 {
                    let shape = shapes.trailing_zeros();
                    shapes &= shapes - 1;
                    values.push((code, shape));
                    enumerate(domains, values, visits, accept)?;
                    values.pop();
                }
            }
        }
    }
    Ok(())
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
                moves: false,
                name: "lhs".into(),
                kind: ParamKind::Value,
            },
            Param {
                moves: false,
                name: "flag".into(),
                kind: ParamKind::Property("bool".into()),
            },
            Param {
                moves: false,
                name: "rhs".into(),
                kind: ParamKind::Value,
            },
        ]
    }

    fn parsed(expression: &str, params: &[Param]) -> Result<Semantic, Error> {
        let source = format!("fixture Test {{ semantics: {expression} }}");
        let mut record = crate::syntax::parse(&source)?.pop().unwrap();
        parse(&source, record.fields.remove("semantics").unwrap(), params)
    }

    #[test]
    fn named_nested_calls_use_logical_value_input_order() {
        let sem = parsed("bv.add(lhs, bv.sub(rhs, bv.one()))", &params()).unwrap();
        assert_eq!(sem.inputs, 2);
        assert_eq!(sem.outputs, [4]);
        assert!(matches!(sem.steps[0], SemanticStep::Input(0)));
        assert!(matches!(sem.steps[1], SemanticStep::Input(1)));
        assert!(matches!(
            sem.steps[2],
            SemanticStep::Const {
                value: BvConst::One,
                ..
            }
        ));
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
            assert!(matches!(sem.steps[0], SemanticStep::Const { value, .. } if value == constant));
        }
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
                moves: false,
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
            moves: false,
            name: "arg".into(),
            kind: ParamKind::Value,
        }];
        let sem = parsed("bv.neg(arg)", &params).unwrap();
        Op {
            offset: 0,
            name: "Test".into(),
            mnemonic: "test".into(),
            meta: crate::model::data::Value::Record("OpInfo".into(), Default::default()),
            format: "Unary".into(),
            signature: TypeDef {
                operands: TypeList::Fixed(vec![operand]),
                results: TypeList::Fixed(vec![result]),
            },
            signature_source: None,
            text: None,
            params,
            projection: crate::model::Projection::Packed(Default::default()),
            traits: vec![],
            memory: crate::model::builtins::Effect::Known(Vec::new()),
            interfaces: Default::default(),
            constraints: vec![],
            identity: None,
            absorbing: None,
            semantics: Some(sem),
        }
    }

    #[test]
    fn same_width_integer_schemes_include_vectors_and_exact_types() {
        for set in ["Integer", "ScalarInteger", "Integer & Vector"] {
            let op = unary(
                Pattern::Bind(0, crate::fixtures::set(set)),
                Pattern::Same(0),
            );
            validate("", &op, &crate::fixtures::types()).unwrap();
        }
        for name in ["I8", "I16", "I32", "I64"] {
            let op = unary(Pattern::Exact(name.into()), Pattern::Exact(name.into()));
            validate("", &op, &crate::fixtures::types()).unwrap();
        }
        for (operand, result) in [
            (Pattern::Exact("BOOL".into()), Pattern::Exact("BOOL".into())),
            (Pattern::Exact("I32".into()), Pattern::Exact("I64".into())),
            (
                Pattern::Bind(0, crate::fixtures::set("Float")),
                Pattern::Same(0),
            ),
            (
                Pattern::Set(crate::fixtures::set("ScalarInteger")),
                Pattern::Set(crate::fixtures::set("ScalarInteger")),
            ),
        ] {
            let op = unary(operand, result);
            assert!(validate("", &op, &crate::fixtures::types()).is_err());
        }
    }

    #[test]
    fn executable_bitvector_semantics_do_not_claim_effects_or_traps() {
        let mut op = unary(Pattern::Exact("I32".into()), Pattern::Exact("I32".into()));
        op.memory = crate::model::builtins::Effect::Unknown;
        assert!(validate("", &op, &crate::fixtures::types()).is_err());
        op.memory = crate::model::builtins::Effect::Known(Vec::new());
        for flag in ["MAY_TRAP", "TERMINATOR"] {
            op.traits = vec![flag.into()];
            assert!(validate("", &op, &crate::fixtures::types()).is_err());
        }
        op.traits.clear();
        op.signature.results = TypeList::Signature;
        assert!(validate("", &op, &crate::fixtures::types()).is_err());
        op.semantics = None;
        validate("", &op, &crate::fixtures::types()).unwrap();
    }
}
