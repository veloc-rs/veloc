//! Compile a named operation signature and its explicit storage projection.

use super::*;
use crate::storage::{FieldType, Format};
use crate::syntax::{Results, Signature};

const PROPERTIES: &[&str] = &[
    "MemFlags",
    "FuncId",
    "SigId",
    "StackSlot",
    "Intrinsic",
    "IntCC",
    "FloatCC",
    "u32",
    "u64",
    "i32",
    "bool",
];

pub(super) fn parse(
    source: &str,
    mut record: Record,
    storage_defs: &storage::Storage,
    type_defs: &Types,
    builtins: &Builtins,
    comparisons: &[crate::comparisons::Comparison],
) -> Result<Op, Error> {
    let sig = record
        .signature
        .take()
        .expect("op parser requires a signature");
    let CheckedSignature {
        params,
        mut types,
        slots,
    } = signature(source, record.offset, sig, storage_defs, type_defs)?;
    let mut fields = Fields::new(source, record);
    let mnemonic_node = fields.take("mnemonic")?;
    let Kind::Text(mnemonic) = mnemonic_node.kind else {
        return Err(fields.error("mnemonic must be a quoted string"));
    };
    let storage = fields.take("storage")?;
    let Kind::Object(format, mappings) = storage.kind else {
        return Err(Error::at(
            source,
            storage.offset,
            "expected a storage mapping",
        ));
    };
    let mut packing = BTreeMap::new();
    for (field, node) in mappings {
        packing.insert(field, binding(source, node)?);
    }
    let signature_source = fields
        .optional("signature")
        .map(|node| match node.kind {
            Kind::Name(name) => Ok(SignatureSource::Signature(name)),
            Kind::Call(kind, mut args) if kind == "function" && args.len() == 1 => {
                Ok(SignatureSource::Function(name(source, args.remove(0))?))
            }
            _ => Err(Error::at(
                source,
                node.offset,
                "expected signature parameter or function(parameter)",
            )),
        })
        .transpose()?;
    let text = fields.optional("text");
    if let Some(node) = fields.optional("where") {
        for node in list(source, node)? {
            let Kind::Call(kind, args) = node.kind else {
                return Err(Error::at(source, node.offset, "expected a type relation"));
            };
            if !matches!(kind.as_str(), "wider" | "narrower" | "same_width_distinct")
                || args.len() != 2
            {
                return Err(Error::at(
                    source,
                    node.offset,
                    "expected type relation wider, narrower or same_width_distinct with two names",
                ));
            }
            let mut resolved = Vec::new();
            for arg in args {
                let offset = arg.offset;
                let name = name(source, arg)?;
                resolved.push(*slots.get(&name).ok_or_else(|| {
                    Error::at(
                        source,
                        offset,
                        format!("type relation references missing operand or result `{name}`"),
                    )
                })?);
            }
            types.relations.push(Relation {
                kind,
                lhs: resolved[0],
                rhs: resolved[1],
            });
        }
    }
    let mut traits = fields
        .optional("traits")
        .map(|n| builtins.traits(source, n))
        .transpose()?
        .unwrap_or_default();
    let constraints = fields
        .optional("constraints")
        .map(|node| list(source, node))
        .transpose()?
        .unwrap_or_default();
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
    let memory = match fields.optional("memory") {
        Some(node) => builtins.effect(source, node)?,
        None if semantics.is_some() => builtins.effect(
            source,
            Node {
                offset: fields.offset,
                kind: Kind::Name("NONE".into()),
            },
        )?,
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
            if !builtins.traits.iter().any(|t| &t.name == name) {
                return Err(
                    fields.error(format!("semantic law requires undeclared trait `{name}`"))
                );
            }
        }
    }
    fields.finish()?;
    let mut op = Op {
        offset: fields.offset,
        name: fields.name,
        mnemonic,
        format,
        signature: types,
        params,
        packing,
        signature_source,
        text,
        traits,
        memory,
        constraints: Vec::new(),
        identity,
        absorbing,
        semantics,
    };
    op.constraints = crate::constraints::check(
        source,
        constraints,
        &op,
        storage_defs,
        type_defs,
        comparisons,
    )?;
    Ok(op)
}

struct CheckedSignature {
    params: Vec<Param>,
    types: TypeDef,
    slots: BTreeMap<String, Slot>,
}

fn signature(
    source: &str,
    offset: usize,
    sig: Signature,
    storage: &storage::Storage,
    types: &Types,
) -> Result<CheckedSignature, Error> {
    let mut variables = BTreeMap::new();
    for generic in sig.generics {
        identifier(source, generic.offset, &generic.name)?;
        if generic.property
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
    let mut names = BTreeSet::new();
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
            if !PROPERTIES.contains(&ty.as_str())
                && ty != "Bytes"
                && !storage.records.iter().any(|record| record.name == ty)
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
        params.push(Param {
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
                patterns.push(pattern(source, result.ty, &mut variables, types)?);
            }
            TypeList::Fixed(patterns)
        }
    };
    for (name, var) in variables {
        if !var.bound {
            return Err(Error::at(
                source,
                offset,
                format!("unused type variable `{name}`"),
            ));
        }
    }
    Ok(CheckedSignature {
        params,
        types: TypeDef {
            operands,
            results,
            relations: Vec::new(),
        },
        slots,
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

pub(super) fn validate_packing(
    source: &str,
    op: &Op,
    format: &Format,
    storage: &storage::Storage,
) -> Result<(), Error> {
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
    for field in op.packing.keys() {
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
            .packing
            .get(&field.name)
            .ok_or_else(|| fail(format!("missing storage field `{}`", field.name)))?;
        match (binding, &field.ty) {
            (Binding::Array(args), FieldType::Values(n) | FieldType::List(n)) => {
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
                        ParamKind::Property(prop) => {
                            (ty == "ConstantPoolId" && prop == "Bytes")
                                || storage
                                    .records
                                    .iter()
                                    .any(|record| record.storage == *ty && record.name == *prop)
                        }
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
            let (param, expected) = match source {
                SignatureSource::Function(param) => (param, "FuncId"),
                SignatureSource::Signature(param) => (param, "SigId"),
            };
            if !matches!(params.get(param.as_str()), Some(ParamKind::Property(ty)) if ty == expected)
            {
                return Err(fail(format!(
                    "signature source `{param}` must be a {expected} property"
                )));
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
