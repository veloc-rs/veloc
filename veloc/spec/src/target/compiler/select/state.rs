//! State identities live only inside a selection recipe. Check their physical
//! lifetimes before encoding them as register constants in selection bytecode.
use super::*;

fn state_unit<'a>(definition: &'a FinalInstDef, operand: &str) -> Option<&'a str> {
    definition
        .state_operands
        .iter()
        .any(|name| name == operand)
        .then(|| {
            definition
                .reg_classes
                .iter()
                .find(|(name, _)| name == operand)
                .unwrap()
                .1[0]
                .as_str()
        })
}

pub(in super::super) fn lower_state_operands(
    rule: &mut SelectRuleDef,
    instructions: &HashMap<String, FinalInstDef>,
) -> Result<(), String> {
    let source = infer_schema_source_def_field(&rule.fields);
    let mut states = HashMap::<String, String>::new();
    // Discover categories first, so ordinary uses before a state definition
    // cannot accidentally turn into reads of a hardware register.
    for constructor in &rule.builds {
        let Constructor::Inst { opcode, args } = constructor else {
            unreachable!()
        };
        let definition = &instructions[opcode];
        let bindings =
            constructor_arg_bindings_by_target_operand(&definition.operands, args, source);
        for (operand, binding) in definition.operands.iter().zip(bindings) {
            let Some(unit) = state_unit(definition, operand.name()) else {
                continue;
            };
            match binding {
                Some(Constructor::Variable(name))
                    if rule.temps.iter().any(|(temp, _)| temp == name) =>
                {
                    if states
                        .insert(name.clone(), unit.into())
                        .is_some_and(|old| old != unit)
                    {
                        return Err(format!(
                            "{opcode}: state temporary {name} has incompatible hardware units"
                        ));
                    }
                }
                Some(Constructor::Reg(reg))
                    if matches!(operand, OperandConstraint::Def(_)) && reg == unit => {}
                _ => {
                    return Err(format!(
                        "{opcode}.{}: state must use a recipe-local temporary; materialize escaping conditions as ordinary values",
                        operand.name()
                    ));
                }
            }
        }
    }
    if states.is_empty() {
        return Ok(());
    }
    let mut defined = BTreeSet::new();
    let mut resident = HashMap::<String, String>::new();
    for constructor in &rule.builds {
        let Constructor::Inst { opcode, args } = constructor else {
            unreachable!()
        };
        let definition = &instructions[opcode];
        let bindings =
            constructor_arg_bindings_by_target_operand(&definition.operands, args, source);
        let mut results = HashMap::new();
        // All inputs observe the old contents, including read/modify/write ops.
        for (operand, binding) in definition.operands.iter().zip(bindings) {
            let Some(Constructor::Variable(name)) = binding else {
                continue;
            };
            let Some(unit) = states.get(name) else {
                continue;
            };
            if state_unit(definition, operand.name()) != Some(unit.as_str()) {
                return Err(format!(
                    "{opcode}.{}: state temporary {name} cannot be used as an ordinary value",
                    operand.name()
                ));
            }
            match operand {
                OperandConstraint::Use(_) if resident.get(unit) != Some(name) => {
                    return Err(format!(
                        "{opcode}: state temporary {name} is undefined or overwritten; materialize the condition before the intervening write"
                    ));
                }
                OperandConstraint::Def(_) => {
                    if !defined.insert(name) || results.insert(unit.clone(), name.clone()).is_some()
                    {
                        return Err(format!(
                            "{opcode}: state results must have unique definitions and hardware units"
                        ));
                    }
                }
                _ => {}
            }
        }
        // Calls may also have ABI-specific clobbers, absent from static metadata.
        if definition.operands.iter().any(|op| {
            matches!(
                op,
                OperandConstraint::Attribute(_, crate::target::AttributeKind::Call)
            )
        }) {
            resident.clear();
        }
        for unit in &definition.clobbers {
            resident.remove(unit);
        }
        for operand in &definition.operands {
            if matches!(operand, OperandConstraint::Def(_)) {
                if let Some((_, registers)) = definition
                    .reg_classes
                    .iter()
                    .find(|(name, _)| name == operand.name())
                {
                    for unit in registers {
                        resident.remove(unit);
                    }
                }
            }
        }
        resident.extend(results);
    }
    // No runtime state values, allocations, residency tracking, or recovery.
    for constructor in &mut rule.builds {
        let Constructor::Inst { args, .. } = constructor else {
            unreachable!()
        };
        for arg in args {
            if let Constructor::Variable(name) = arg {
                if let Some(unit) = states.get(name) {
                    *arg = Constructor::Reg(unit.clone());
                }
            }
        }
    }
    rule.temps.retain(|(name, _)| !states.contains_key(name));
    Ok(())
}
