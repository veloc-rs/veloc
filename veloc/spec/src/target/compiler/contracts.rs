//! Resolve OpSpec contracts against a target's registers and encoding bodies.
use super::{FinalInstDef, assembly};
use crate::target::{AttributeKind, Def, Module, OperandConstraint};
use crate::{
    schema::{Contract, Operand, ValueTypes},
    syntax::{Kind, Node},
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Target-owned constant schemas. OpSpec parses typed constants and preserves
/// references; this layer resolves machine identities and checks their meaning.
pub(super) fn registers(declarations: &[crate::syntax::Decl]) -> Result<Vec<Def>, String> {
    use crate::syntax::DeclKind;
    use crate::target::{RegClassDef, RegDef, RegisterAlias, RegisterWrite};
    let mut roots = BTreeMap::new();
    let mut views = BTreeMap::new();
    let mut classes = Vec::new();
    let mut names = BTreeSet::new();
    for declaration in declarations {
        let DeclKind::Constant {
            ty,
            value: Some(value),
        } = &declaration.kind
        else {
            continue;
        };
        let Kind::Name(schema) = &ty.kind else {
            continue;
        };
        if !matches!(
            schema.as_str(),
            "Register" | "RegisterView" | "RegisterClass"
        ) {
            continue;
        }
        if !names.insert(declaration.name.clone()) {
            return Err(format!("duplicate machine constant `{}`", declaration.name));
        }
        let Kind::Object(constructor, properties) = &value.kind else {
            return Err(format!("{} requires a {schema} literal", declaration.name));
        };
        if constructor != schema {
            return Err(format!(
                "{}: declared type and constructor differ",
                declaration.name
            ));
        }
        let mut properties = properties.clone();
        let mut required = |field: &str| {
            properties
                .remove(field)
                .ok_or_else(|| format!("{}: missing {schema}.{field}", declaration.name))
        };
        match schema.as_str() {
            "Register" => {
                let size = number(&required("bits")?)?;
                let id = number(&required("id")?)?;
                let hw_enc = number(&required("encoding")?)?;
                let reserved = match name(&required("reserved")?)? {
                    "true" => true,
                    "false" => false,
                    _ => return Err("Register.reserved requires a boolean".into()),
                };
                let roles = list(Some(required("roles")?))?
                    .into_iter()
                    .map(|role| match role.as_str() {
                        "RegisterRole::StackPointer" => Ok("stack-pointer".into()),
                        "RegisterRole::FramePointer" => Ok("frame-pointer".into()),
                        _ => Err(format!("unknown register role `{role}`")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if size == 0
                    || size > u16::MAX as u32
                    || id > u16::MAX as u32
                    || hw_enc > u16::MAX as u32
                {
                    return Err("register size or encoding out of range".into());
                }
                roots.insert(
                    declaration.name.clone(),
                    RegDef {
                        name: declaration.name.clone(),
                        size,
                        id,
                        hw_enc,
                        reserved,
                        roles,
                        alias: None,
                        state_type: properties
                            .remove("state_type")
                            .map(|node| name(&node).map(str::to_owned))
                            .transpose()?,
                    },
                );
            }
            "RegisterView" => {
                let base = name(&required("base")?)?.to_owned();
                let offset = number(&required("offset")?)?;
                let size = number(&required("bits")?)?;
                let write = match name(&required("write")?)? {
                    "WriteEffect::Preserve" => RegisterWrite::Preserve,
                    "WriteEffect::ZeroExtend" => RegisterWrite::ZeroExtend,
                    _ => return Err("unknown register write effect".into()),
                };
                views.insert(
                    declaration.name.clone(),
                    (
                        size,
                        RegisterAlias {
                            base,
                            offset,
                            write,
                        },
                    ),
                );
            }
            "RegisterClass" => {
                let regs = list(Some(required("members")?))?;
                if regs.is_empty() || regs.iter().collect::<BTreeSet<_>>().len() != regs.len() {
                    return Err("register class requires distinct members".into());
                }
                classes.push(RegClassDef {
                    name: declaration.name.clone(),
                    regs,
                });
            }
            _ => unreachable!(),
        }
        finish(&properties)?;
    }
    let mut encodings = BTreeSet::new();
    let mut roles = BTreeSet::new();
    for root in roots.values() {
        if !encodings.insert(root.id) {
            return Err("physical register identities must be unique".into());
        }
        for role in &root.roles {
            if !roles.insert(role.clone()) {
                return Err(format!("duplicate register role `{role}`"));
            }
        }
    }
    // Views refer directly to a storage root, making overlap and write effects
    // unambiguous. Chains would otherwise need composition of write effects.
    let mut resolved = Vec::new();
    for (name, (size, alias)) in views {
        let base = roots
            .get(&alias.base)
            .ok_or_else(|| format!("{name}: register view must reference a declared root"))?;
        if size == 0
            || alias
                .offset
                .checked_add(size)
                .is_none_or(|end| end > base.size)
        {
            return Err(format!("{name}: register view lies outside its root"));
        }
        if alias.write == RegisterWrite::ZeroExtend && alias.offset != 0 {
            return Err(format!(
                "{name}: zero-extending view must start at bit zero"
            ));
        }
        resolved.push(RegDef {
            name,
            size,
            id: base.id,
            hw_enc: base.hw_enc,
            reserved: base.reserved,
            roles: Vec::new(),
            alias: Some(alias),
            state_type: None,
        });
    }
    let register_names: BTreeSet<_> = roots
        .keys()
        .chain(resolved.iter().map(|reg| &reg.name))
        .collect();
    for class in &classes {
        if class.regs.iter().any(|reg| !register_names.contains(reg)) {
            return Err(format!("{}: undeclared register class member", class.name));
        }
    }
    let mut result: Vec<_> = roots.into_values().chain(resolved).map(Def::Reg).collect();
    result.extend(classes.into_iter().map(Def::RegClass));
    Ok(result)
}

fn name(node: &Node) -> Result<&str, String> {
    match &node.kind {
        Kind::Name(name) => Ok(name),
        _ => Err("expected a name".into()),
    }
}
fn object(node: Node) -> Result<BTreeMap<String, Node>, String> {
    match node.kind {
        Kind::Record(fields) => Ok(fields),
        _ => Err("expected an anonymous object literal".into()),
    }
}
fn number(node: &Node) -> Result<u32, String> {
    match node.kind {
        Kind::Number(n) => Ok(n),
        _ => Err("expected a u32 integer".into()),
    }
}
fn list(node: Option<Node>) -> Result<Vec<String>, String> {
    match node {
        None => Ok(Vec::new()),
        Some(Node {
            kind: Kind::List(nodes),
            ..
        }) => nodes.iter().map(|n| name(n).map(str::to_owned)).collect(),
        _ => Err("expected a name list".into()),
    }
}
fn finish(fields: &BTreeMap<String, Node>) -> Result<(), String> {
    match fields.first_key_value() {
        Some((name, _)) => Err(format!("unknown field `{name}`")),
        None => Ok(()),
    }
}

pub(super) fn compile(
    contracts: Vec<Contract>,
    module: &Module,
    types: &crate::types::Types,
) -> Result<HashMap<String, FinalInstDef>, String> {
    let regs: BTreeSet<_> = module
        .defs
        .iter()
        .filter_map(|d| {
            if let Def::Reg(r) = d {
                Some(r.name.clone())
            } else {
                None
            }
        })
        .collect();
    let features: BTreeSet<_> = module
        .defs
        .iter()
        .filter_map(|d| match d {
            Def::Feature(feature) => Some(feature.name.as_str()),
            _ => None,
        })
        .collect();
    let mut classes = BTreeMap::new();
    for def in &module.defs {
        match def {
            Def::RegClass(class) => {
                if class.regs.is_empty() || class.regs.iter().any(|r| !regs.contains(r)) {
                    return Err(format!(
                        "{}: register class must contain declared registers",
                        class.name
                    ));
                }
                if classes
                    .insert(class.name.clone(), class.regs.clone())
                    .is_some()
                {
                    return Err(format!("duplicate register class {}", class.name));
                }
            }
            _ => {}
        }
    }
    let mut result = HashMap::new();
    // Preserve the declared category identity: different state categories may
    // share the same logical representation (for example, boolean flags).
    let mut state_types = BTreeMap::new();
    for def in &module.defs {
        let Def::Reg(reg) = def else { continue };
        let Some(category) = &reg.state_type else {
            continue;
        };
        let ty = types
            .exact
            .get(category)
            .ok_or_else(|| format!("{}: unknown state type `{category}`", reg.name))?;
        if !reg.reserved
            || reg.alias.is_some()
            || !ty.is_singleton()
            || !ty.0.iter().all(|(primitive, shapes)| {
                *shapes == 1 && primitive.element_bits() == Some(reg.size)
            })
        {
            return Err(format!(
                "{}: state type requires a reserved root with matching scalar width",
                reg.name
            ));
        }
        let category = category.strip_prefix("Type::").unwrap_or(category);
        if state_types
            .insert(category.to_owned(), reg.name.clone())
            .is_some()
        {
            return Err(format!(
                "state type `{category}` has multiple storage roots"
            ));
        }
    }
    for contract in contracts {
        let op = contract.name.clone();
        let build = || -> Result<FinalInstDef, String> {
            if !matches!(contract.signature.operands, ValueTypes::Fixed(_))
                || !matches!(contract.signature.results, ValueTypes::Fixed(_))
            {
                return Err(
                    "machine instructions require fixed input and result signatures".into(),
                );
            }
            let mut operands: Vec<_> = contract
                .results
                .iter()
                .cloned()
                .map(OperandConstraint::Def)
                .collect();
            for input in contract.inputs {
                operands.push(match input {
                    Operand::Value(n) => OperandConstraint::Use(n),
                    Operand::Attribute { name, ty } => OperandConstraint::Attribute(
                        name,
                        AttributeKind::from_spec(&ty)
                            .ok_or_else(|| format!("unsupported machine attribute type `{ty}`"))?,
                    ),
                    _ => return Err("machine instructions require fixed operands".into()),
                });
            }
            let names = operands
                .iter()
                .filter_map(|operand| match operand {
                    OperandConstraint::Use(name) => Some(name.clone()),
                    _ => None,
                })
                .chain(contract.results.iter().cloned());
            let patterns = contract
                .signature
                .operands
                .patterns()
                .unwrap()
                .iter()
                .chain(contract.signature.results.patterns().unwrap());
            let mut state_locations = BTreeMap::new();
            let value_types = names
                .zip(patterns)
                .map(|(name, pattern)| {
                    if let crate::model::Pattern::Exact(ty) = pattern
                        && let Some(reg) = state_types.get(ty.strip_prefix("Type::").unwrap_or(ty))
                    {
                        state_locations.insert(name.clone(), reg.clone());
                    }
                    let set = match pattern {
                        crate::model::Pattern::Set(set) => set.clone(),
                        crate::model::Pattern::Exact(ty) => types.exact[ty].clone(),
                        _ => {
                            return Err(
                                "machine operand representations must be explicit type sets"
                                    .to_owned(),
                            );
                        }
                    };
                    Ok((name, set))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut fields = contract.fields;
            let requires = match fields.remove("requires") {
                None => Vec::new(),
                Some(Node {
                    kind: Kind::List(nodes),
                    ..
                }) => nodes
                    .iter()
                    .map(|node| {
                        let Kind::Name(feature) = &node.kind else {
                            return Err(
                                "requires expects references to declared features".to_owned()
                            );
                        };
                        if !features.contains(feature.as_str()) {
                            return Err(format!("unknown target feature {feature}"));
                        }
                        Ok(feature.clone())
                    })
                    .collect::<Result<Vec<_>, String>>()?,
                _ => return Err("requires expects a feature list".into()),
            };
            let mut locations = fields
                .remove("registers")
                .map(object)
                .transpose()?
                .unwrap_or_default();
            let mut reg_classes = Vec::new();
            let mut state_operands = Vec::new();
            let mut ties = Vec::new();
            for index in 0..operands.len() {
                let (operand, is_result) = match &operands[index] {
                    OperandConstraint::Def(n) => (n.clone(), true),
                    OperandConstraint::Use(n) => (n.clone(), false),
                    _ => continue,
                };
                if let Some(reg) = state_locations.get(&operand) {
                    if locations.contains_key(&operand) {
                        return Err(format!(
                            "`{operand}` already has a placement from its state type"
                        ));
                    }
                    state_operands.push(operand.clone());
                    reg_classes.push((operand, vec![reg.clone()]));
                    continue;
                }
                let node = locations
                    .remove(&operand)
                    .ok_or_else(|| format!("missing register constraint for `{operand}`"))?;
                let (class, relation) = match &node.kind {
                    Kind::Name(class) => (class.as_str(), None),
                    Kind::Call(kind, args)
                        if matches!(kind.as_str(), "fixed" | "tied") && args.len() == 2 =>
                    {
                        (name(&args[1])?, Some((kind.as_str(), name(&args[0])?)))
                    }
                    _ => return Err(
                        "expected register class, fixed(register, class), or tied(input, class)"
                            .into(),
                    ),
                };
                let mut allowed = classes
                    .get(class)
                    .ok_or_else(|| format!("unknown register class `{class}`"))?
                    .clone();
                if let Some((kind, target)) = relation {
                    if kind == "fixed" {
                        if !allowed.iter().any(|r| r == target) {
                            return Err(format!("{target} is not in {class}"));
                        }
                        allowed.retain(|r| r == target);
                    } else {
                        if !is_result {
                            return Err("tied constraints belong to results".into());
                        }
                        let input = operands
                            .iter()
                            .position(|o| matches!(o, OperandConstraint::Use(n) if n == target))
                            .ok_or_else(|| format!("tied input `{target}` does not exist"))?;
                        ties.push((index, input));
                    }
                }
                reg_classes.push((operand, allowed));
            }
            finish(&locations)?;
            for &(dst, src) in &ties {
                let def = operands[dst].name();
                let input = operands[src].name();
                let a = &reg_classes.iter().find(|(n, _)| n == def).unwrap().1;
                let b = &reg_classes.iter().find(|(n, _)| n == input).unwrap().1;
                if !a.iter().any(|r| b.contains(r)) {
                    return Err(format!(
                        "tied operands {def} and {input} have disjoint register constraints"
                    ));
                }
            }
            let clobbers = list(fields.remove("clobbers"))?;
            for reg in &clobbers {
                if !regs.contains(reg) {
                    return Err(format!("unknown clobbered register {reg}"));
                }
            }
            // State results install values; clobbers only destroy old contents.
            let mut state_results = BTreeSet::new();
            for operand in &operands {
                let OperandConstraint::Def(name) = operand else {
                    continue;
                };
                let Some(reg) = state_locations.get(name) else {
                    continue;
                };
                if !state_results.insert(reg) {
                    return Err(format!("multiple results occupy state register `{reg}`"));
                }
                if clobbers.contains(reg) {
                    return Err(format!(
                        "state result `{name}` already describes the write to `{reg}`"
                    ));
                }
            }
            let schedule_class = fields
                .remove("schedule")
                .map(|node| match node.kind {
                    Kind::Name(name) if name == "None" => Ok(None),
                    Kind::Name(class) => Ok(Some(class)),
                    _ => Err(
                        "schedule must reference a declared scheduling class or None".to_owned(),
                    ),
                })
                .transpose()?
                .flatten();
            let flow = fields
                .remove("flow")
                .map(|n| name(&n).map(str::to_owned))
                .transpose()?
                .unwrap_or_else(|| "Next".into());
            if !matches!(
                flow.as_str(),
                "Next" | "Jump" | "Branch" | "Return" | "Call" | "Trap"
            ) {
                return Err(format!("unknown control flow `{flow}`"));
            }
            let has_memory = if let Some(node) = fields.remove("memory") {
                let mut fields = object(node)?;
                let kind = fields.remove("kind").ok_or("missing memory kind")?;
                let bytes = number(&fields.remove("bytes").ok_or("missing memory size")?)?;
                if !matches!(name(&kind)?, "Read" | "Write") || bytes == 0 {
                    return Err("expected Read/Write with a positive byte size".into());
                }
                finish(&fields)?;
                true
            } else {
                false
            };
            let flag_count = operands
                .iter()
                .filter(|op| matches!(op, OperandConstraint::Attribute(_, AttributeKind::MemFlags)))
                .count();
            if flag_count != usize::from(has_memory) {
                return Err("a memory instruction requires exactly one MemFlags field; other instructions must not have one".into());
            }
            let is_pseudo = match fields.remove("pseudo") {
                None => false,
                Some(node) if name(&node)? == "true" => true,
                _ => return Err("pseudo must be true when specified".into()),
            };
            if is_pseudo && schedule_class.is_some() {
                return Err("pseudo instruction costs must be assigned after expansion".into());
            }
            if !is_pseudo && schedule_class.is_none() {
                return Err("machine instruction requires a schedule class".into());
            }
            // Permission to reorder is independent of the CPU cost model.
            // An absent declaration remains conservative, even with a cost.
            let movable = match fields.remove("movable") {
                None => false,
                Some(node) => match name(&node)? {
                    "true" => true,
                    "false" => false,
                    _ => return Err("movable requires a boolean".into()),
                },
            };
            let rematerializable = match fields.remove("rematerializable") {
                None => false,
                Some(node) if name(&node)? == "true" => true,
                _ => return Err("rematerializable must be true when specified".into()),
            };
            if rematerializable && (has_memory || flow != "Next" || is_pseudo) {
                return Err("state rematerialization requires a pure producer".into());
            }
            // Checked by the shared expression compiler after operand resolution.
            fields.remove("encoding");
            let assembly = fields.remove("assembly");
            finish(&fields)?;
            let mut inst = FinalInstDef {
                operands,
                reg_classes,
                state_operands,
                rematerializable,
                value_types,
                ties,
                clobbers,
                schedule_class,
                movable,
                flow,
                encoding: None,
                is_pseudo,
                assembly: None,
                requires,
            };
            inst.assembly = assembly
                .as_ref()
                .map(|node| assembly::compile(node, &inst))
                .transpose()?;
            if !inst.is_pseudo && inst.assembly.is_none() {
                return Err("instruction requires an assembly field".into());
            }
            Ok(inst)
        };
        let inst = build().map_err(|e| format!("{op}: {e}"))?;
        if result.insert(op.clone(), inst).is_some() {
            return Err(format!("duplicate instruction `{op}`"));
        }
    }
    Ok(result)
}
