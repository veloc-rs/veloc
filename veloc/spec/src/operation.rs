//! Compile a named operation signature and its explicit storage projection.

use super::*;
use crate::storage::{FieldType, Format};
use crate::syntax::{Results, Signature};

const PROPERTIES: &[&str] = &[
    "VectorExtId",
    "VectorMemExtId",
    "MemFlags",
    "FuncId",
    "SigId",
    "StackSlot",
    "PtrIndexImmId",
    "ConstantPoolId",
    "Intrinsic",
    "IntCC",
    "FloatCC",
    "u32",
    "u64",
    "i32",
    "bool",
];

pub(super) fn parse(source: &str, mut record: Record) -> Result<Op, Error> {
    let sig = record
        .signature
        .take()
        .expect("op parser requires a signature");
    let CheckedSignature {
        params,
        mut types,
        slots,
    } = signature(source, record.offset, sig)?;
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
        let values = match node.kind {
            Kind::List(nodes) => nodes
                .into_iter()
                .map(|n| name(source, n))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![name(source, node)?],
        };
        packing.insert(field, values);
    }
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
        .map(|n| choices(source, n, TRAITS, "trait"))
        .transpose()?
        .unwrap_or_default();
    let constraints = fields
        .optional("constraints")
        .map(|node| {
            list(source, node)?
                .into_iter()
                .map(|n| name(source, n))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let mut identity = fields
        .optional("identity")
        .map(|n| choice(source, n, CONSTANTS, "algebraic constant"))
        .transpose()?;
    let mut absorbing = fields
        .optional("absorbing")
        .map(|n| choice(source, n, CONSTANTS, "algebraic constant"))
        .transpose()?;
    let semantics = fields
        .optional("semantics")
        .map(|node| crate::semantic::parse(source, node, &params))
        .transpose()?;
    let memory = match fields.optional("memory") {
        Some(node) => choice(source, node, MEMORY, "memory effect")?,
        None if semantics.is_some() => "NONE".into(),
        None => return Err(fields.error("unmodeled operations must declare their memory effect")),
    };
    if let Some(semantics) = &semantics {
        crate::semantic::derive(
            source,
            fields.offset,
            semantics,
            &mut traits,
            &mut identity,
            &mut absorbing,
        )?;
    }
    fields.finish()?;
    Ok(Op {
        offset: fields.offset,
        name: fields.name,
        mnemonic,
        format,
        signature: types,
        params,
        packing,
        traits,
        memory,
        constraints,
        identity,
        absorbing,
        semantics,
    })
}

struct CheckedSignature {
    params: Vec<Param>,
    types: TypeDef,
    slots: BTreeMap<String, Slot>,
}

fn signature(source: &str, offset: usize, sig: Signature) -> Result<CheckedSignature, Error> {
    let mut variables = BTreeMap::new();
    for generic in sig.generics {
        identifier(source, generic.offset, &generic.name)?;
        if generic.property
            || CLASSES.contains(&generic.name.as_str())
            || EXACT.contains(&generic.name.as_str())
        {
            return Err(Error::at(
                source,
                generic.offset,
                "invalid type variable declaration",
            ));
        }
        let class = choice(source, generic.ty, CLASSES, "type class")?;
        let slot = u8::try_from(variables.len())
            .map_err(|_| Error::at(source, generic.offset, "more than 256 type variables"))?;
        if variables
            .insert(
                generic.name.clone(),
                Variable {
                    slot,
                    domain: class_domain(&class),
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
            ParamKind::Property(choice(source, param.ty, PROPERTIES, "property type")?)
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
            patterns.push(pattern(source, param.ty, &mut variables)?);
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
                identifier(source, result.offset, &result.name)?;
                if result.property {
                    return Err(Error::at(
                        source,
                        result.offset,
                        "results cannot be properties",
                    ));
                }
                if !names.insert(result.name.clone()) {
                    return Err(Error::at(
                        source,
                        result.offset,
                        format!("duplicate parameter or result `{}`", result.name),
                    ));
                }
                let index = u8::try_from(patterns.len())
                    .map_err(|_| Error::at(source, result.offset, "more than 256 results"))?;
                slots.insert(
                    result.name,
                    Slot {
                        result: true,
                        index,
                    },
                );
                patterns.push(pattern(source, result.ty, &mut variables)?);
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

pub(super) fn validate_packing(source: &str, op: &Op, format: &Format) -> Result<(), Error> {
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
    for field in &format.fields {
        if matches!(&field.ty, FieldType::Named(ty) if ty == "Opcode") {
            continue;
        }
        let args = op
            .packing
            .get(&field.name)
            .ok_or_else(|| fail(format!("missing storage field `{}`", field.name)))?;
        let expected = match field.ty {
            FieldType::Values(n) | FieldType::List(n) => n,
            _ => 1,
        };
        if args.len() != expected {
            return Err(fail(format!(
                "storage field `{}` requires {expected} arguments",
                field.name
            )));
        }
        for arg in args {
            let kind = params
                .get(arg.as_str())
                .ok_or_else(|| fail(format!("unknown parameter `{arg}` in storage mapping")))?;
            if !used.insert(arg.as_str()) {
                return Err(fail(format!("parameter `{arg}` is stored more than once")));
            }
            let compatible = match (&field.ty, kind) {
                (FieldType::Values(_) | FieldType::List(_), ParamKind::Value) => true,
                (FieldType::Named(ty), ParamKind::Value) => ty == "Value",
                (FieldType::Named(ty), ParamKind::Values) => ty == "ValueList",
                (FieldType::Named(ty), ParamKind::Successor) => ty == "BlockCall",
                (FieldType::Named(ty), ParamKind::Successors) => ty == "JumpTable",
                (FieldType::Named(ty), ParamKind::Property(prop)) => ty == prop,
                _ => false,
            };
            if !compatible {
                return Err(fail(format!(
                    "parameter `{arg}` is incompatible with storage field `{}`",
                    field.name
                )));
            }
            if !matches!(kind, ParamKind::Property(_)) {
                operands.push(arg.as_str());
            }
        }
    }
    for param in &op.params {
        if !used.contains(param.name.as_str()) {
            return Err(fail(format!(
                "parameter `{}` has no storage mapping",
                param.name
            )));
        }
    }
    // Custom MIR consumers currently share a flat operand ABI. Properties
    // can be reordered independently, but operand permutations need adapters.
    let logical: Vec<_> = op
        .params
        .iter()
        .filter(|p| !matches!(p.kind, ParamKind::Property(_)))
        .map(|p| p.name.as_str())
        .collect();
    if logical != operands {
        return Err(fail(
            "storage mapping changes logical operand order; an operand adapter is required".into(),
        ));
    }
    Ok(())
}
