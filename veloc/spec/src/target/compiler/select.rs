use crate::target::ast::{CondCode, Constructor, DeclDef, Def, Pattern, SelectRuleDef};
use crate::target::{ExtractorDef, OperandConstraint, PatternArg};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

use super::FinalInstDef;
mod matcher;
use super::generate::{collect_reg_ids, sanitize_ident};

fn positional_arg_at(args: &[PatternArg], index: usize) -> Option<&Pattern> {
    args.iter()
        .filter_map(|arg| match arg {
            PatternArg::Positional(pattern) => Some(pattern),
            PatternArg::Named { .. } => None,
        })
        .nth(index)
}

fn named_args(args: &[PatternArg]) -> impl Iterator<Item = (&str, &Pattern)> {
    args.iter().filter_map(|arg| match arg {
        PatternArg::Named { name, pattern } => Some((name.as_str(), pattern.as_ref())),
        PatternArg::Positional(_) => None,
    })
}

fn collect_decl_map(module: &crate::target::ast::Module) -> HashMap<String, DeclDef> {
    let mut decls = HashMap::new();
    for def in &module.defs {
        if let Def::Decl(decl) = def {
            decls.insert(decl.name.clone(), decl.clone());
        }
    }
    decls
}

/// Extractors are pure predicate compositions. Their names carry no built-in
/// meaning; every leaf must satisfy an explicitly declared host signature.
pub(super) fn check_predicates(module: &crate::target::ast::Module) -> Result<(), String> {
    fn check(
        pattern: &Pattern,
        locals: &[String],
        decls: &HashMap<String, DeclDef>,
    ) -> Result<(), String> {
        match pattern {
            Pattern::And(parts) => {
                for part in parts {
                    check(part, locals, decls)?;
                }
            }
            Pattern::Opcode { opcode, args, .. } if opcode == "not" => {
                let [PatternArg::Positional(inner)] = args.as_slice() else {
                    return Err("not requires exactly one predicate".into());
                };
                check(inner, locals, decls)?;
            }
            Pattern::Opcode { opcode, args, .. } => {
                let decl = decls
                    .get(opcode)
                    .ok_or_else(|| format!("undeclared selector predicate {opcode}"))?;
                if args.len() != decl.params.len() {
                    return Err(format!("predicate {opcode} has incorrect argument count"));
                }
                for arg in args {
                    let PatternArg::Positional(Pattern::Variable(name)) = arg else {
                        return Err(format!("predicate {opcode} requires a bound register"));
                    };
                    if !locals.contains(name) {
                        return Err(format!("unbound extractor parameter {name}"));
                    }
                }
            }
            _ => return Err("extractor body must be a predicate composition".into()),
        }
        Ok(())
    }
    let decls = collect_decl_map(module);
    for decl in decls.values() {
        if decl.params.len() > 1 || decl.params.iter().any(|ty| ty != "vreg") {
            return Err(format!(
                "predicate {} supports zero or one vreg argument",
                decl.name
            ));
        }
    }
    for def in &module.defs {
        if let Def::Extractor(extractor) = def {
            if extractor.args.len() > 1 {
                return Err("extractor supports at most one value parameter".into());
            }
            check(&extractor.body, &extractor.args, &decls)?;
        }
    }
    Ok(())
}

fn collect_select_rules_by_opcode<'a>(
    module: &'a crate::target::ast::Module,
) -> HashMap<String, Vec<&'a SelectRuleDef>> {
    let mut opcode_rules: HashMap<String, Vec<&SelectRuleDef>> = HashMap::new();
    for def in &module.defs {
        if let Def::SelectRule(rule) = def {
            opcode_rules
                .entry(rule.opcode.clone())
                .or_default()
                .push(rule);
        }
    }
    opcode_rules
}

fn generate_pattern_condition(
    pattern: &Pattern,
    var_name: &str,
    decls: &HashMap<String, DeclDef>,
) -> String {
    match pattern {
        Pattern::NodeBind { inner, .. } => generate_pattern_condition(inner, var_name, decls),
        Pattern::And(pats) => {
            let conds: Vec<String> = pats
                .iter()
                .map(|p| generate_pattern_condition(p, var_name, decls))
                .filter(|cond| cond != "true")
                .collect();
            match conds.len() {
                0 => "true".to_string(),
                1 => conds.into_iter().next().unwrap(),
                _ => conds.join(" && "),
            }
        }
        Pattern::Schema { .. } => "true".to_string(),
        Pattern::Opcode { opcode, args, .. } => match opcode.as_str() {
            "not" => positional_arg_at(args, 0)
                .map(|arg| format!("!({})", generate_pattern_condition(arg, var_name, decls)))
                .unwrap_or_else(|| "true".to_string()),
            _ => {
                let decl = decls.get(opcode).expect("checked predicate declaration");
                if decl.params.is_empty() {
                    format!("ctx.{}()", opcode)
                } else {
                    format!("ctx.{}({})", opcode, var_name)
                }
            }
        },
        _ => "true".to_string(),
    }
}

fn render_cond_code_match(schema_name: &str, cc: CondCode) -> Option<&'static str> {
    match schema_name {
        "ICmp" => Some(match cc {
            CondCode::E => "veloc_mir::IntCC::Eq",
            CondCode::NE => "veloc_mir::IntCC::Ne",
            CondCode::L => "veloc_mir::IntCC::LtS",
            CondCode::LE => "veloc_mir::IntCC::LeS",
            CondCode::G => "veloc_mir::IntCC::GtS",
            CondCode::GE => "veloc_mir::IntCC::GeS",
            CondCode::B => "veloc_mir::IntCC::LtU",
            CondCode::BE => "veloc_mir::IntCC::LeU",
            CondCode::A => "veloc_mir::IntCC::GtU",
            CondCode::AE => "veloc_mir::IntCC::GeU",
        }),
        "FCmp" => Some(match cc {
            CondCode::E => "veloc_mir::FloatCC::Eq",
            CondCode::NE => "veloc_mir::FloatCC::Ne",
            CondCode::L | CondCode::B => "veloc_mir::FloatCC::Lt",
            CondCode::LE | CondCode::BE => "veloc_mir::FloatCC::Le",
            CondCode::G | CondCode::A => "veloc_mir::FloatCC::Gt",
            CondCode::GE | CondCode::AE => "veloc_mir::FloatCC::Ge",
        }),
        _ => None,
    }
}

fn collect_pattern_variables(pattern: &Pattern, vars: &mut Vec<String>) {
    match pattern {
        Pattern::NodeBind { inner, .. } => collect_pattern_variables(inner, vars),
        Pattern::Variable(name) | Pattern::Typed { name, .. } => vars.push(name.clone()),
        Pattern::Opcode { args, .. } | Pattern::Schema { args, .. } => {
            for arg in args {
                match arg {
                    PatternArg::Positional(pattern) => collect_pattern_variables(pattern, vars),
                    PatternArg::Named { pattern, .. } => collect_pattern_variables(pattern, vars),
                }
            }
        }
        Pattern::And(parts) => {
            for part in parts {
                collect_pattern_variables(part, vars);
            }
        }
        _ => {}
    }
}

fn min_explicit_args_from(operands: &[OperandConstraint], start: usize) -> usize {
    operands[start..]
        .iter()
        .filter(|op| !matches!(op, OperandConstraint::Def(_)))
        .count()
}

pub(crate) fn infer_schema_source_def_field(args: &[PatternArg]) -> Option<&str> {
    if named_args(args).any(|(field, _)| field == "dst") {
        Some("dst")
    } else {
        None
    }
}

fn collect_field_variable_bindings(args: &[PatternArg]) -> HashMap<String, String> {
    let mut bindings = HashMap::new();
    for (field, pattern) in named_args(args) {
        let mut vars = Vec::new();
        collect_pattern_variables(pattern, &mut vars);
        for var in vars {
            bindings.entry(var).or_insert_with(|| field.to_string());
        }
    }
    bindings
}

fn constructor_arg_bindings_by_target_operand<'a>(
    target_operands: &[OperandConstraint],
    constructor_args: &'a [Constructor],
    schema_source_def_field: Option<&str>,
) -> Vec<Option<&'a Constructor>> {
    let mut explicit_arg_cursor = 0usize;
    let mut bindings = Vec::with_capacity(target_operands.len());

    for (operand_index, operand) in target_operands.iter().enumerate() {
        let use_source_def = matches!(operand, OperandConstraint::Def(_))
            && schema_source_def_field.is_some()
            && constructor_args.len() - explicit_arg_cursor
                == min_explicit_args_from(target_operands, operand_index);

        if use_source_def {
            bindings.push(None);
            continue;
        }

        bindings.push(constructor_args.get(explicit_arg_cursor));
        explicit_arg_cursor += 1;
    }

    bindings
}

/// Omitted fixed-state results are fresh dead SSA values, never missing machine
/// effects. Explicitly supplied results can be connected to other constructors.
pub(super) fn complete_state_results(
    rule: &mut SelectRuleDef,
    instructions: &HashMap<String, FinalInstDef>,
    types: &crate::types::Types,
) -> Result<(), String> {
    let source = infer_schema_source_def_field(&rule.fields);
    let fields = collect_field_variable_bindings(&rule.fields);
    let mut source = fields
        .iter()
        .find(|(_, field)| Some(field.as_str()) == source)
        .map(|(value, _)| value.clone());
    let single = rule.builds.len() == 1;
    for (build_index, ctor) in rule.builds.iter_mut().enumerate() {
        let Constructor::Inst { opcode, args } = ctor else {
            continue;
        };
        let Some(definition) = instructions.get(opcode) else {
            continue;
        };
        if args.len() == definition.operands.len() {
            continue;
        }
        let omitted = definition.operands.iter().filter(|op| matches!(op, OperandConstraint::Def(n) if definition.state_operands.contains(n))).count();
        if omitted == 0 {
            continue;
        }
        let remaining = definition.operands.len() - omitted;
        let implicit_result = single && args.len() + 1 == remaining;
        if args.len() != remaining && !implicit_result {
            continue;
        }
        if implicit_result && source.is_none() {
            let name = "__state_source_result".to_owned();
            rule.fields.push(PatternArg::Named {
                name: "dst".into(),
                pattern: Box::new(Pattern::Variable(name.clone())),
            });
            source = Some(name);
        }
        let mut original = core::mem::take(args).into_iter();
        let mut source = implicit_result.then(|| source.clone().unwrap());
        for op in &definition.operands {
            if let OperandConstraint::Def(name) = op {
                if definition.state_operands.contains(name) {
                    let fresh = format!("__state_{build_index}_{name}");
                    if rule.temps.iter().any(|(name, _)| name == &fresh)
                        || fields.contains_key(&fresh)
                    {
                        return Err("reserved state temporary name used in selection".into());
                    }
                    let domain = &definition
                        .value_types
                        .iter()
                        .find(|(n, _)| n == name)
                        .unwrap()
                        .1;
                    let scalar = types
                        .scalars
                        .iter()
                        .find(|scalar| &types.exact[&scalar.name] == domain)
                        .ok_or_else(|| {
                            format!(
                                "{opcode}: state result `{name}` requires a scalar representation"
                            )
                        })?;
                    rule.temps
                        .push((fresh.clone(), format!("Type::{}", scalar.name)));
                    args.push(Constructor::Variable(fresh));
                    continue;
                }
                if let Some(value) = source.take() {
                    args.push(Constructor::Variable(value));
                    continue;
                }
            }
            args.push(original.next().ok_or("missing constructor operand")?);
        }
    }
    Ok(())
}
pub(super) fn check_construction(
    rule: &SelectRuleDef,
    instructions: &HashMap<String, FinalInstDef>,
    types: &crate::types::Types,
) -> Result<(), String> {
    fn domain(names: &[String], types: &crate::types::Types) -> crate::types::TypeSet {
        let mut set = crate::types::TypeSet::default();
        for name in names {
            set.union(&types.exact[name.rsplit("::").next().unwrap()]);
        }
        set
    }
    fn collect(
        pattern: &Pattern,
        types: &crate::types::Types,
        known: &mut HashMap<String, crate::types::TypeSet>,
    ) {
        match pattern {
            Pattern::Typed { name, types: names } => {
                let set = domain(names, types);
                known
                    .entry(name.clone())
                    .and_modify(|old| old.intersect(&set))
                    .or_insert(set);
            }
            Pattern::And(parts) => {
                for part in parts {
                    collect(part, types, known);
                }
            }
            _ => {}
        }
    }
    let mut known = HashMap::new();
    for (_, pattern) in named_args(&rule.fields) {
        collect(pattern, types, &mut known);
    }
    for (name, ty) in &rule.temps {
        known.insert(name.clone(), domain(std::slice::from_ref(ty), types));
    }
    let sequence = rule.builds.len() > 1;
    for ctor in &rule.builds {
        let Constructor::Inst { opcode, args } = ctor else {
            return Err("build requires an instruction constructor".into());
        };
        let definition = instructions
            .get(opcode)
            .ok_or_else(|| format!("unknown target instruction {opcode}"))?;
        let count = definition.operands.len();
        let required = if sequence {
            count
        } else {
            min_explicit_args_from(&definition.operands, 0)
        };
        if args.len() < required || args.len() > count {
            return Err(format!(
                "{opcode} expects {required}..={count} operands, got {}",
                args.len()
            ));
        }
        if args
            .iter()
            .any(|arg| matches!(arg, Constructor::Inst { .. }))
        {
            return Err("nested instruction constructors require separate build operations".into());
        }
        let source = infer_schema_source_def_field(&rule.fields);
        let bindings =
            constructor_arg_bindings_by_target_operand(&definition.operands, args, source);
        let fields = collect_field_variable_bindings(&rule.fields);
        for (operand, binding) in definition.operands.iter().zip(bindings) {
            let name = match operand {
                OperandConstraint::Def(name) | OperandConstraint::Use(name) => name,
                _ => continue,
            };
            let variable = match binding {
                Some(Constructor::Variable(value)) => Some(value.as_str()),
                None => fields
                    .iter()
                    .find(|(_, field)| Some(field.as_str()) == source)
                    .map(|(value, _)| value.as_str()),
                _ => None,
            };
            if let Some(actual) = variable.and_then(|value| known.get(value)) {
                let expected = &definition
                    .value_types
                    .iter()
                    .find(|(field, _)| field == name)
                    .unwrap()
                    .1;
                if !actual.is_empty() && !actual.subset_of(expected) {
                    return Err(format!(
                        "{}: {opcode}.{name} does not accept the representation of {}",
                        rule.opcode,
                        variable.unwrap()
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn check_temps(rule: &crate::target::ast::SelectRuleDef) -> Result<(), String> {
    let mut bindings = collect_field_variable_bindings(&rule.fields);
    for (name, _) in &rule.temps {
        if bindings.insert(name.clone(), String::new()).is_some() {
            return Err(format!("temporary {name} must be fresh"));
        }
    }
    Ok(())
}

pub(super) fn check_storage(
    rule: &SelectRuleDef,
    layouts: &BTreeMap<String, crate::storage::operands::Projection>,
) -> Result<(), String> {
    let paths = named_args(&rule.fields).map(|(path, _)| path).chain(
        rule.definitions
            .iter()
            .map(|definition| definition.input.as_str()),
    );
    for path in paths {
        let (opcode, field) = if let Some((owner, field)) = path.split_once('.') {
            let def = rule
                .definitions
                .iter()
                .find(|def| def.name == owner)
                .ok_or_else(|| format!("unknown definition {owner}"))?;
            (&def.opcode, field)
        } else {
            (&rule.opcode, path)
        };
        let member = layouts
            .get(opcode)
            .and_then(|layout| layout.members.iter().find(|m| m.field.name == field))
            .ok_or_else(|| format!("{opcode}.{field} has no operand storage projection"))?;
        if member.binding.is_none() {
            return Err(format!(
                "{opcode}.{field}: field is fixed to none and has no readable storage position"
            ));
        }
        if member.field.shape == crate::storage::operands::Shape::Sequence {
            return Err(format!(
                "{opcode}.{field}: sequence fields require a sequence selection operation, not a scalar access"
            ));
        }
    }
    Ok(())
}

pub(crate) fn generate_select_instruction(
    output: &mut String,
    module: &crate::target::ast::Module,
    extractors: &HashMap<String, ExtractorDef>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    _arch: &str,
    context: Option<&str>,
    layouts: &BTreeMap<String, crate::storage::operands::Projection>,
) {
    let regs = collect_reg_ids(module);
    let decls = collect_decl_map(module);
    let rules = collect_select_rules_by_opcode(module);
    let mut adapters = matcher::Adapters::new(layouts);
    let mut opcodes: Vec<_> = rules.keys().collect();
    opcodes.sort();
    let groups: Vec<_> = opcodes
        .iter()
        .map(|opcode| rules[*opcode].as_slice())
        .collect();
    matcher::emit(
        output,
        &groups,
        extractors,
        final_inst_defs,
        &regs,
        &mut adapters,
    );
    adapters.emit(output, context, extractors, &decls);
}
