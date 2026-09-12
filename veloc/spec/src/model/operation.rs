//! Compile a named operation signature and its explicit storage projection.

use super::*;
use crate::storage::{FieldType, Format};
use crate::syntax::{Results, Signature};

pub(super) fn parse(
    source: &str,
    mut record: Record,
    storage_defs: &storage::Storage,
    vocabulary: Vocabulary<'_>,
    expressions: &mut super::expr::Library,
) -> Result<Op, Error> {
    let Vocabulary { builtins, data, .. } = vocabulary;
    let sig = record
        .signature
        .take()
        .expect("op parser requires a signature");
    let CheckedSignature {
        params,
        types,
        slots,
        type_bindings,
    } = signature(source, record.offset, sig, vocabulary)?;
    let mut fields = Fields::new(source, record);
    let mnemonic = match fields.optional("mnemonic") {
        Some(Node {
            kind: Kind::Text(name),
            ..
        }) => name,
        Some(_) => return Err(fields.error("mnemonic must be a quoted string")),
        None => match &storage_defs.strategy {
            storage::Strategy::Operands(operands) => operands.mnemonic(&fields.name),
            storage::Strategy::Packed => return Err(fields.error("missing field `mnemonic`")),
        },
    };
    let storage = fields.take("storage")?;
    let Kind::Object(format, mappings) = storage.kind else {
        return Err(Error::at(
            source,
            storage.offset,
            "expected storage: Layout { field mappings }",
        ));
    };
    let projection = match &storage_defs.strategy {
        storage::Strategy::Packed => {
            let mut packing = BTreeMap::new();
            for (field, node) in mappings {
                packing.insert(field, binding(source, node)?);
            }
            Projection::Packed(packing)
        }
        storage::Strategy::Operands(operands) => {
            let flow = fields.optional("flow");
            Projection::Operands(operands.project(
                source,
                storage.offset,
                &format,
                &mappings,
                &params,
                &types,
                &slots,
                flow,
            )?)
        }
    };
    let signature_source = fields
        .optional("signature")
        .map(|node| match node.kind {
            Kind::Name(name) => Ok(SignatureSource::Signature(name)),
            Kind::Call(kind, mut args) if kind == "function" && args.len() == 1 => {
                Ok(SignatureSource::Function(name(source, args.remove(0))?))
            }
            Kind::Call(kind, mut args) if kind == "callable" && args.len() == 1 => {
                Ok(SignatureSource::Value(name(source, args.remove(0))?))
            }
            _ => Err(Error::at(
                source,
                node.offset,
                "expected signature parameter, function(parameter) or callable(parameter)",
            )),
        })
        .transpose()?;
    let text = fields.optional("text");
    let implementations = expressions.bind(
        source,
        fields.optional("implements"),
        &params,
        &types,
        vocabulary,
    )?;
    let control = fields
        .optional("control")
        .map(|node| crate::model::control::check(source, node, &params))
        .transpose()?;
    let mut meta_node = fields.take("meta")?;
    expressions.metadata(source, &mut meta_node, &implementations, vocabulary)?;
    let meta = crate::model::metadata::Pending::new(source, meta_node, data)?;
    let mut traits = meta.traits(source, data, builtins)?;
    if let Projection::Operands(projection) = &projection {
        let required: &[&str] = match projection.flow.as_str() {
            "Jump" | "Return" => &["TERMINATOR"],
            "Trap" => &["TERMINATOR", "ABORT", "MAY_TRAP"],
            "Call" => &["MAY_TRAP"],
            _ => &[],
        };
        for &name in required {
            if !builtins.has_trait(name) {
                return Err(fields.error(format!("flow requires undeclared trait `{name}`")));
            }
            if !traits.iter().any(|t| t == name) {
                traits.push(name.into());
            }
        }
    }
    let constraints = fields
        .optional("verify")
        .map(|node| list(source, node))
        .transpose()?
        .unwrap_or_default();
    if let Some(control) = &control {
        for &name in control.traits() {
            if !builtins.has_trait(name) {
                return Err(fields.error(format!(
                    "control interface requires undeclared trait `{name}`"
                )));
            }
            if !traits.iter().any(|t| t == name) {
                traits.push(name.into());
            }
        }
    }
    if traits.iter().any(|t| t == "ABORT") && !traits.iter().any(|t| t == "TERMINATOR") {
        return Err(fields.error("ABORT requires TERMINATOR"));
    }
    let mut identity = fields
        .optional("identity")
        .map(|n| algebraic_constant(source, n))
        .transpose()?;
    let mut absorbing = fields
        .optional("absorbing")
        .map(|n| algebraic_constant(source, n))
        .transpose()?;
    let mut semantics = fields
        .optional("semantics")
        .map(|node| crate::semantic::parse(source, node, &params))
        .transpose()?;
    if let Some(node) = fields.optional("traps") {
        let sem = semantics
            .as_mut()
            .ok_or_else(|| fields.error("trap guards require executable semantics"))?;
        crate::semantic::traps(source, node, &params, sem)?;
    }
    let declared_memory = meta.memory(source, data, builtins)?;
    let memory = match declared_memory {
        _ if semantics.is_some() && !meta.explicit_memory() => {
            crate::model::builtins::Effect::Known(Vec::new())
        }
        Some(memory) => memory,
        None if semantics.is_some() => crate::model::builtins::Effect::Known(Vec::new()),
        None if !meta.has_memory_field() => crate::model::builtins::Effect::Unknown,
        None => return Err(fields.error("unmodeled operations must declare their memory effect")),
    };
    if let Some(semantics) = &semantics {
        if !semantics.traps.is_empty() && !traits.iter().any(|t| t == "MAY_TRAP") {
            traits.push("MAY_TRAP".into());
        }
        crate::semantic::derive(
            source,
            fields.offset,
            semantics,
            &mut traits,
            &mut identity,
            &mut absorbing,
        )?;
        for name in &traits {
            if !builtins.has_trait(name) {
                return Err(
                    fields.error(format!("semantic law requires undeclared trait `{name}`"))
                );
            }
        }
    }
    fields.finish()?;
    let meta = meta.finish(source, data, builtins, &traits, &memory)?;
    let mut op = Op {
        offset: fields.offset,
        name: fields.name,
        mnemonic,
        meta,
        format,
        signature: types,
        params,
        projection,
        signature_source,
        control,
        text,
        traits,
        memory,
        interfaces: implementations,
        constraints: Vec::new(),
        identity,
        absorbing,
        semantics,
    };
    op.constraints = expressions.verify(
        source,
        constraints,
        &op.params,
        Some(&op.signature),
        &type_bindings,
        vocabulary,
    )?;
    Ok(op)
}

struct CheckedSignature {
    params: Vec<Param>,
    types: TypeDef,
    slots: BTreeMap<String, Slot>,
    type_bindings: BTreeMap<String, Slot>,
}

fn signature(
    source: &str,
    offset: usize,
    sig: Signature,
    vocabulary: Vocabulary<'_>,
) -> Result<CheckedSignature, Error> {
    let Vocabulary { types, data, .. } = vocabulary;
    let mut variables = BTreeMap::new();
    for generic in sig.generics {
        identifier(source, generic.offset, &generic.name)?;
        if generic.moves
            || generic.property
            || types.classes.contains_key(&generic.name)
            || types.exact.contains_key(&generic.name)
        {
            return Err(Error::at(
                source,
                generic.offset,
                "invalid type variable declaration",
            ));
        }
        let class = types.set(source, &generic.ty)?;
        let slot = u8::try_from(variables.len())
            .map_err(|_| Error::at(source, generic.offset, "more than 256 type variables"))?;
        if variables
            .insert(
                generic.name.clone(),
                Variable {
                    slot,
                    possible: class.clone(),
                    class,
                    bound: false,
                },
            )
            .is_some()
        {
            return Err(Error::at(
                source,
                generic.offset,
                format!("duplicate type variable `{}`", generic.name),
            ));
        }
    }
    let mut params = Vec::new();
    let mut patterns = Vec::new();
    let mut names = variables.keys().cloned().collect::<BTreeSet<_>>();
    let mut slots = BTreeMap::new();
    let mut variadic = false;
    for param in sig.params {
        identifier(source, param.offset, &param.name)?;
        if !names.insert(param.name.clone()) {
            return Err(Error::at(
                source,
                param.offset,
                format!("duplicate parameter `{}`", param.name),
            ));
        }
        let kind = if param.property {
            let offset = param.ty.offset;
            let ty = name(source, param.ty)?;
            if !super::records::primitive(&ty)
                && ty != "Bytes"
                && (!data.names.contains(&ty) || ty == "Value")
            {
                return Err(Error::at(
                    source,
                    offset,
                    format!("unknown property type `{ty}`"),
                ));
            }
            ParamKind::Property(ty)
        } else if let Kind::Name(kind) = &param.ty.kind
            && matches!(kind.as_str(), "values" | "successor" | "successors")
        {
            variadic = true;
            match kind.as_str() {
                "values" => ParamKind::Values,
                "successor" => ParamKind::Successor,
                "successors" => ParamKind::Successors,
                _ => unreachable!(),
            }
        } else {
            if variadic {
                return Err(Error::at(
                    source,
                    param.offset,
                    "fixed operands must precede variadic operands and successors",
                ));
            }
            let index = u8::try_from(patterns.len())
                .map_err(|_| Error::at(source, param.offset, "more than 256 operands"))?;
            slots.insert(
                param.name.clone(),
                Slot {
                    result: false,
                    index,
                },
            );
            patterns.push(pattern(source, param.ty, &mut variables, types)?);
            ParamKind::Value
        };
        if param.moves && !matches!(kind, ParamKind::Value | ParamKind::Values) {
            return Err(Error::at(
                source,
                param.offset,
                "move requires a value operand; successor arguments transfer on their edge",
            ));
        }
        params.push(Param {
            moves: param.moves,
            name: param.name,
            kind,
        });
    }
    let operands = if variadic {
        TypeList::Variadic(patterns)
    } else {
        TypeList::Fixed(patterns)
    };
    let results = match sig.results {
        Results::Signature => TypeList::Signature,
        Results::Fixed(results) => {
            let mut patterns = Vec::new();
            for result in results {
                let index = u8::try_from(patterns.len())
                    .map_err(|_| Error::at(source, result.offset, "more than 256 results"))?;
                if let Some(name) = result.name {
                    identifier(source, result.offset, &name)?;
                    if !names.insert(name.clone()) {
                        return Err(Error::at(
                            source,
                            result.offset,
                            format!("duplicate parameter or result `{name}`"),
                        ));
                    }
                    slots.insert(
                        name,
                        Slot {
                            result: true,
                            index,
                        },
                    );
                }
                let pat = if let Kind::Call(kind, args) = &result.ty.kind
                    && kind == "type"
                {
                    let [
                        Node {
                            kind: Kind::Name(name),
                            ..
                        },
                    ] = args.as_slice()
                    else {
                        return Err(Error::at(
                            source,
                            result.offset,
                            "type expects a typed property parameter",
                        ));
                    };
                    let param = params.iter().find(|p| p.name == *name);
                    let set = param
                        .and_then(|p| match &p.kind {
                            ParamKind::Property(ty) => types.property_types(ty),
                            _ => None,
                        })
                        .ok_or_else(|| {
                            Error::at(
                                source,
                                result.offset,
                                "type expects a typed property parameter",
                            )
                        })?;
                    Pattern::Property(name.clone(), set)
                } else {
                    pattern(source, result.ty, &mut variables, types)?
                };
                patterns.push(pat);
            }
            TypeList::Fixed(patterns)
        }
    };
    let mut type_bindings = slots
        .iter()
        .filter(|(_, s)| s.result)
        .map(|(n, s)| (n.clone(), *s))
        .collect::<BTreeMap<_, _>>();
    for (name, var) in variables {
        if !var.bound {
            return Err(Error::at(
                source,
                offset,
                format!("unused type variable `{name}`"),
            ));
        }
        let slot = [(false, &operands), (true, &results)]
            .into_iter()
            .find_map(|(result, list)| {
                list.patterns()?
                    .iter()
                    .position(|p| matches!(p, Pattern::Bind(id, _) if *id == var.slot))
                    .map(|index| Slot {
                        result,
                        index: index as u8,
                    })
            })
            .expect("bound generic has a defining type slot");
        type_bindings.insert(name, slot);
    }
    Ok(CheckedSignature {
        params,
        types: TypeDef { operands, results },
        slots,
        type_bindings,
    })
}

fn binding(source: &str, node: Node) -> Result<Binding, Error> {
    match node.kind {
        Kind::Name(name) => Ok(Binding::Name(name)),
        Kind::List(nodes) => nodes
            .into_iter()
            .map(|node| binding(source, node))
            .collect::<Result<Vec<_>, _>>()
            .map(Binding::Array),
        Kind::Call(kind, mut args) if kind == "pool" && args.len() == 1 => {
            Ok(Binding::Pool(name(source, args.remove(0))?))
        }
        Kind::Call(kind, mut args) if kind == "table" && args.len() == 2 => {
            let cases = name(source, args.remove(0))?;
            let default = name(source, args.remove(0))?;
            Ok(Binding::Table { cases, default })
        }
        _ => Err(Error::at(
            source,
            node.offset,
            "expected parameter, array, pool(parameter) or table(cases, default)",
        )),
    }
}

pub(super) fn validate_packing(source: &str, op: &Op, format: &Format) -> Result<(), Error> {
    if let Some(control) = &op.control {
        control.validate(source, op)?;
    }
    let fail = |message| Error::at(source, op.offset, message);
    let params: BTreeMap<_, _> = op
        .params
        .iter()
        .map(|p| (p.name.as_str(), &p.kind))
        .collect();
    let mut used = BTreeSet::new();
    let mut operands = Vec::new();
    let fields: BTreeSet<_> = format
        .fields
        .iter()
        .filter(|f| !matches!(&f.ty, FieldType::Named(ty) if ty == "Opcode"))
        .map(|f| f.name.as_str())
        .collect();
    for field in op.bindings().keys() {
        if !fields.contains(field.as_str()) {
            return Err(fail(format!(
                "unknown storage field `{field}` in `{}`",
                op.format
            )));
        }
    }
    let mut use_param =
        |arg: &str, compatible: &dyn Fn(&ParamKind) -> bool, field: &str| -> Result<(), Error> {
            let kind = params
                .get(arg)
                .ok_or_else(|| fail(format!("unknown parameter `{arg}` in storage mapping")))?;
            if !used.insert(arg.to_owned()) {
                return Err(fail(format!("parameter `{arg}` is stored more than once")));
            }
            if !compatible(kind) {
                return Err(fail(format!(
                    "parameter `{arg}` is incompatible with storage field `{field}`"
                )));
            }
            if !matches!(kind, ParamKind::Property(_)) {
                operands.push(arg.to_owned());
            }
            Ok(())
        };
    for field in &format.fields {
        if matches!(&field.ty, FieldType::Named(ty) if ty == "Opcode") {
            continue;
        }
        let binding = op
            .bindings()
            .get(&field.name)
            .ok_or_else(|| fail(format!("missing storage field `{}`", field.name)))?;
        match (binding, &field.ty) {
            (Binding::Array(args), FieldType::Values(n)) => {
                if args.len() != *n {
                    return Err(fail(format!(
                        "storage field `{}` requires {n} arguments",
                        field.name
                    )));
                }
                for arg in args {
                    let Binding::Name(arg) = arg else {
                        return Err(fail("array storage accepts only value parameters".into()));
                    };
                    use_param(arg, &|kind| matches!(kind, ParamKind::Value), &field.name)?;
                }
            }
            (Binding::Name(arg), FieldType::Named(ty)) => {
                use_param(
                    arg,
                    &|kind| match kind {
                        ParamKind::Value => ty == "Value",
                        ParamKind::Values => ty == "ValueList",
                        ParamKind::Successor => ty == "BlockCall",
                        ParamKind::Successors => false,
                        ParamKind::Property(prop) => ty == prop,
                    },
                    &field.name,
                )?;
            }
            (Binding::Pool(arg), FieldType::Named(ty)) => {
                use_param(
                    arg,
                    &|kind| match kind {
                        ParamKind::Property(prop) => ty == "ConstantPoolId" && prop == "Bytes",
                        _ => false,
                    },
                    &field.name,
                )?;
            }
            (Binding::Table { cases, default }, FieldType::Named(ty)) if ty == "JumpTable" => {
                use_param(
                    cases,
                    &|kind| matches!(kind, ParamKind::Successors),
                    &field.name,
                )?;
                use_param(
                    default,
                    &|kind| matches!(kind, ParamKind::Successor),
                    &field.name,
                )?;
            }
            _ => {
                return Err(fail(format!(
                    "binding is incompatible with storage field `{}`",
                    field.name
                )));
            }
        }
    }
    for param in &op.params {
        if !used.contains(&param.name) {
            return Err(fail(format!(
                "parameter `{}` has no storage mapping",
                param.name
            )));
        }
    }
    let logical: Vec<_> = op
        .params
        .iter()
        .filter(|p| !matches!(p.kind, ParamKind::Property(_)))
        .map(|p| p.name.clone())
        .collect();
    if logical != operands {
        return Err(fail(
            "storage mapping changes logical operand order; an operand adapter is required".into(),
        ));
    }
    match (&op.signature_source, &op.signature.results) {
        (None, TypeList::Signature) => {
            return Err(fail(
                "signature results require an explicit signature source".into(),
            ));
        }
        (Some(source), TypeList::Signature) => {
            match source {
                SignatureSource::Value(param) => {
                    let index = op
                        .params
                        .iter()
                        .filter(|p| p.kind == ParamKind::Value)
                        .position(|p| p.name == *param);
                    let patterns = match &op.signature.operands {
                        TypeList::Fixed(patterns) | TypeList::Variadic(patterns) => patterns,
                        TypeList::Signature => {
                            unreachable!("operands cannot use signature results")
                        }
                    };
                    if index.and_then(|i| patterns.get(i)) != Some(&Pattern::Callable) {
                        return Err(fail(format!(
                            "signature source `{param}` must be a Callable value operand"
                        )));
                    }
                    if !op
                        .bindings()
                        .values()
                        .any(|binding| matches!(binding, Binding::Name(name) if name == param))
                    {
                        return Err(fail(format!(
                            "signature source `{param}` requires direct value storage"
                        )));
                    }
                }
                SignatureSource::Function(param) | SignatureSource::Signature(param) => {
                    let expected = if matches!(source, SignatureSource::Function(_)) {
                        "FuncId"
                    } else {
                        "SigId"
                    };
                    if !matches!(params.get(param.as_str()), Some(ParamKind::Property(ty)) if ty == expected)
                    {
                        return Err(fail(format!(
                            "signature source `{param}` must be a {expected} property"
                        )));
                    }
                }
            }
            if op
                .params
                .iter()
                .filter(|p| p.kind == ParamKind::Values)
                .count()
                != 1
            {
                return Err(fail(
                    "signature operation requires exactly one values parameter".into(),
                ));
            }
        }
        (Some(_), _) => return Err(fail("signature source requires signature results".into())),
        (None, _) => {}
    }
    Ok(())
}
