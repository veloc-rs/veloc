use crate::target::ast::{CondCode, Constructor, DeclDef, Def, Pattern, SelectRuleDef};
use crate::target::{ExtractorDef, OperandConstraint, PatternArg};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

use super::FinalInstDef;
use super::generate::{collect_reg_ids, format_slice, reg_const_name, sanitize_ident};

fn strip_node_binds(pattern: &Pattern) -> &Pattern {
    pattern.strip_node_binds()
}

fn pattern_opcode(pattern: &Pattern) -> Option<&str> {
    match strip_node_binds(pattern) {
        Pattern::Opcode { opcode, .. } => Some(opcode.as_str()),
        Pattern::Schema { opcode, .. } => Some(opcode.as_str()),
        _ => None,
    }
}

fn pattern_args(pattern: &Pattern) -> Option<&[PatternArg]> {
    match strip_node_binds(pattern) {
        Pattern::Opcode { args, .. } => Some(args.as_slice()),
        Pattern::Schema { args, .. } => Some(args.as_slice()),
        _ => None,
    }
}

fn pattern_schema_name(pattern: &Pattern) -> Option<&str> {
    match strip_node_binds(pattern) {
        Pattern::Schema { schema, .. } => Some(schema.as_str()),
        _ => None,
    }
}

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

pub(crate) fn module_has_positional_rules(module: &crate::target::ast::Module) -> bool {
    module.defs.iter().any(|def| {
        let Def::SelectRule(rule) = def else {
            return false;
        };
        let Some(pattern) = rule.patterns.first() else {
            return false;
        };
        pattern_schema_name(pattern).is_none()
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

fn collect_select_rules_by_opcode<'a>(
    module: &'a crate::target::ast::Module,
) -> HashMap<String, Vec<&'a SelectRuleDef>> {
    let mut opcode_rules: HashMap<String, Vec<&SelectRuleDef>> = HashMap::new();
    for def in &module.defs {
        if let Def::SelectRule(rule) = def {
            if let Some(pattern) = rule.patterns.first() {
                if let Some(opcode) = pattern_opcode(pattern) {
                    opcode_rules
                        .entry(opcode.to_string())
                        .or_default()
                        .push(rule);
                }
            }
        }
    }
    opcode_rules
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BindingSource {
    OperandIndex(usize),
    SchemaValue,
}

struct InstEmitContext<'a> {
    var_map: &'a HashMap<String, BindingSource>,
    final_inst_defs: &'a HashMap<String, FinalInstDef>,
    reg_map: &'a HashMap<String, u32>,
    schema_var: Option<&'a str>,
    schema_source_def_field: Option<&'a str>,
}

struct InstEmitRequest {
    index: usize,
    preserve_operands: bool,
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
            "is_i8" => format!("ctx.is_i8({})", var_name),
            "is_i16" => format!("ctx.is_i16({})", var_name),
            "is_i32" => format!("ctx.is_i32({})", var_name),
            "is_i64" => format!("ctx.is_i64({})", var_name),
            "is_ptr" => format!("ctx.is_ptr({})", var_name),
            "is_fpr" => format!("ctx.is_fpr({})", var_name),
            "has_bmi2" => "ctx.has_bmi2()".to_string(),
            "has_avx2" => "ctx.has_avx2()".to_string(),
            "not" => positional_arg_at(args, 0)
                .map(|arg| format!("!({})", generate_pattern_condition(arg, var_name, decls)))
                .unwrap_or_else(|| "true".to_string()),
            _ => {
                if let Some(decl) = decls.get(opcode) {
                    if decl.params.is_empty() {
                        format!("ctx.{}()", opcode)
                    } else {
                        format!("ctx.{}({})", opcode, var_name)
                    }
                } else {
                    format!("ctx.{}()", opcode)
                }
            }
        },
        _ => "true".to_string(),
    }
}

fn pattern_condition_needs_value(pattern: &Pattern, decls: &HashMap<String, DeclDef>) -> bool {
    match strip_node_binds(pattern) {
        Pattern::And(parts) => parts
            .iter()
            .any(|part| pattern_condition_needs_value(part, decls)),
        Pattern::Schema { .. }
        | Pattern::Variable(_)
        | Pattern::IntConst(_)
        | Pattern::CondCode(_)
        | Pattern::StackSlot(_)
        | Pattern::Block(_) => false,
        Pattern::Opcode { opcode, args, .. } => match opcode.as_str() {
            "is_i8" | "is_i16" | "is_i32" | "is_i64" | "is_ptr" | "is_fpr" => true,
            "has_bmi2" | "has_avx2" => false,
            "not" => positional_arg_at(args, 0)
                .map(|arg| pattern_condition_needs_value(arg, decls))
                .unwrap_or(false),
            _ => decls
                .get(opcode)
                .is_some_and(|decl| !decl.params.is_empty()),
        },
        Pattern::NodeBind { .. } => unreachable!("node binds are stripped above"),
    }
}

fn collect_used_extractors_in_pattern(
    pattern: &Pattern,
    extractors: &HashMap<String, ExtractorDef>,
    used: &mut BTreeSet<String>,
) {
    match strip_node_binds(pattern) {
        Pattern::Schema { args, .. } | Pattern::Opcode { args, .. } => {
            if let Pattern::Opcode { opcode, .. } = strip_node_binds(pattern) {
                if let Some(extractor) = extractors.get(opcode) {
                    if used.insert(opcode.clone()) {
                        collect_used_extractors_in_pattern(&extractor.body, extractors, used);
                    }
                }
            }

            for arg in args {
                match arg {
                    PatternArg::Positional(pattern) => {
                        collect_used_extractors_in_pattern(pattern, extractors, used);
                    }
                    PatternArg::Named { pattern, .. } => {
                        collect_used_extractors_in_pattern(pattern, extractors, used);
                    }
                }
            }
        }
        Pattern::And(parts) => {
            for part in parts {
                collect_used_extractors_in_pattern(part, extractors, used);
            }
        }
        Pattern::Variable(_)
        | Pattern::IntConst(_)
        | Pattern::CondCode(_)
        | Pattern::StackSlot(_)
        | Pattern::Block(_) => {}
        Pattern::NodeBind { .. } => unreachable!("node binds are stripped above"),
    }
}

fn collect_used_extractors(
    module: &crate::target::ast::Module,
    extractors: &HashMap<String, ExtractorDef>,
) -> BTreeSet<String> {
    let mut used = BTreeSet::new();
    for def in &module.defs {
        let Def::SelectRule(rule) = def else {
            continue;
        };
        for pattern in &rule.patterns {
            collect_used_extractors_in_pattern(pattern, extractors, &mut used);
        }
    }
    used
}

fn collect_positional_rule_conditions(
    args: &[PatternArg],
    extractors: &HashMap<String, ExtractorDef>,
) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter_map(|(i, arg)| {
            let PatternArg::Positional(pattern) = arg else {
                return None;
            };
            if let Pattern::Opcode {
                opcode: exc_name, ..
            } = strip_node_binds(pattern)
            {
                if extractors.contains_key(exc_name) {
                    Some(format!("is_{}(v{})", exc_name.to_lowercase(), i))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

fn collect_schema_rule_conditions(
    args: &[PatternArg],
    extractors: &HashMap<String, ExtractorDef>,
    schema_var: &str,
    schema_name: &str,
) -> Vec<String> {
    named_args(args)
        .filter_map(|(field, pattern)| {
            collect_schema_field_conditions(field, pattern, extractors, schema_var, schema_name)
                .map(|parts| parts.join(" && "))
        })
        .filter(|cond| !cond.is_empty())
        .collect()
}

fn collect_schema_field_conditions(
    field: &str,
    pattern: &Pattern,
    extractors: &HashMap<String, ExtractorDef>,
    schema_var: &str,
    schema_name: &str,
) -> Option<Vec<String>> {
    match strip_node_binds(pattern) {
        Pattern::Variable(_) => None,
        Pattern::And(parts) => {
            let conds: Vec<String> = parts
                .iter()
                .filter_map(|part| {
                    collect_schema_field_conditions(
                        field,
                        part,
                        extractors,
                        schema_var,
                        schema_name,
                    )
                })
                .flatten()
                .collect();
            if conds.is_empty() { None } else { Some(conds) }
        }
        Pattern::Opcode { opcode, .. } if extractors.contains_key(opcode) => Some(vec![format!(
            "is_{}(reg_value_to_vreg({}.{}))",
            opcode.to_lowercase(),
            schema_var,
            field
        )]),
        Pattern::IntConst(value) => Some(vec![format!("{}.{} == {}", schema_var, field, value)]),
        Pattern::CondCode(cc) => render_cond_code_match(schema_name, *cc)
            .map(|expr| vec![format!("{}.{} == {}", schema_var, field, expr)]),
        Pattern::Block(_) => None,
        Pattern::StackSlot(_) | Pattern::Opcode { .. } => None,
        Pattern::Schema { .. } => None,
        Pattern::NodeBind { .. } => None,
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

fn collect_var_bindings(pattern: &Pattern) -> HashMap<String, BindingSource> {
    let mut map = HashMap::new();
    if let Some(args) = pattern_args(pattern) {
        if pattern_schema_name(pattern).is_some() {
            for (_field, arg) in named_args(args) {
                collect_vars_in_pattern(arg, BindingSource::SchemaValue, &mut map);
            }
        } else {
            for (idx, arg) in args.iter().enumerate() {
                let PatternArg::Positional(arg) = arg else {
                    continue;
                };
                collect_vars_in_pattern(arg, BindingSource::OperandIndex(idx), &mut map);
            }
        }
    }
    map
}

fn emit_schema_value_bindings_for_group(
    output: &mut String,
    rules: &[&SelectRuleDef],
    schema_var: &str,
) {
    let mut needed_vars = BTreeSet::new();
    for rule in rules {
        collect_constructor_variables(&rule.emit, &mut needed_vars);
    }

    let mut bindings = BTreeMap::new();
    for rule in rules {
        let Some(pattern) = rule.patterns.first() else {
            continue;
        };
        let Some(args) = pattern_args(pattern) else {
            continue;
        };
        for (field, pattern) in named_args(args) {
            let mut vars = Vec::new();
            collect_pattern_variables(pattern, &mut vars);
            for var in vars {
                if needed_vars.contains(&var) {
                    bindings.entry(var).or_insert_with(|| field.to_string());
                }
            }
        }
    }

    for (var, field) in bindings {
        let rust_var = rust_ident(&var);
        writeln!(
            output,
            "                let {} = {}.{}.clone();",
            rust_var, schema_var, field
        )
        .unwrap();
    }
}

fn collect_pattern_variables(pattern: &Pattern, vars: &mut Vec<String>) {
    match pattern {
        Pattern::NodeBind { inner, .. } => collect_pattern_variables(inner, vars),
        Pattern::Variable(name) => vars.push(name.clone()),
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

fn collect_constructor_variables(constructor: &Constructor, vars: &mut BTreeSet<String>) {
    match constructor {
        Constructor::Inst { args, .. } => {
            for arg in args {
                collect_constructor_variables(arg, vars);
            }
        }
        Constructor::Variable(name) => {
            vars.insert(name.clone());
        }
        Constructor::Imm(_) | Constructor::Reg(_) => {}
    }
}

fn schema_group_needs_binding(
    rules: &[&SelectRuleDef],
    extractors: &HashMap<String, ExtractorDef>,
) -> bool {
    let mut needed_vars = BTreeSet::new();
    for rule in rules {
        collect_constructor_variables(&rule.emit, &mut needed_vars);
    }

    for rule in rules {
        let Some(pattern) = rule.patterns.first() else {
            continue;
        };
        let Some(args) = pattern_args(pattern) else {
            continue;
        };

        if infer_schema_source_def_field(args).is_some() {
            return true;
        }
        if let Some(schema_name) = pattern_schema_name(pattern) {
            if !collect_schema_rule_conditions(args, extractors, "schema_inst", schema_name)
                .is_empty()
            {
                return true;
            }
        }
        for (_field, pattern) in named_args(args) {
            let mut vars = Vec::new();
            collect_pattern_variables(pattern, &mut vars);
            if vars.into_iter().any(|var| needed_vars.contains(&var)) {
                return true;
            }
        }
    }

    false
}

fn collect_vars_in_pattern(
    pat: &Pattern,
    source: BindingSource,
    map: &mut HashMap<String, BindingSource>,
) {
    match pat {
        Pattern::NodeBind { inner, .. } => collect_vars_in_pattern(inner, source, map),
        Pattern::Variable(name) => {
            map.insert(name.clone(), source);
        }
        Pattern::Opcode { args, .. } => {
            for arg in args {
                match arg {
                    PatternArg::Positional(arg) => {
                        collect_vars_in_pattern(arg, source.clone(), map);
                    }
                    PatternArg::Named { pattern, .. } => {
                        collect_vars_in_pattern(pattern, source.clone(), map);
                    }
                }
            }
        }
        Pattern::And(list) => {
            for p in list {
                collect_vars_in_pattern(p, source.clone(), map);
            }
        }
        _ => {}
    }
}

fn emit_constructor_sequence(
    output: &mut String,
    constructor: &Constructor,
    ctx: &InstEmitContext<'_>,
    request: InstEmitRequest,
) {
    // Decode every operand before the first store write. This also makes a
    // failed match side-effect free and ends source-view borrows before emission.
    let mut commits = String::new();
    match constructor {
        Constructor::Inst { opcode, args } if opcode == "seq" => {
            for (i, c) in args.iter().enumerate() {
                emit_single_inst(
                    output,
                    &mut commits,
                    c,
                    ctx,
                    InstEmitRequest {
                        index: i,
                        preserve_operands: false,
                    },
                );
            }
            output.push_str(&commits);
            writeln!(output, "                return Ok(SelectResult::Replace);").unwrap();
        }
        _ => {
            emit_single_inst(output, &mut commits, constructor, ctx, request);
            output.push_str(&commits);
            writeln!(output, "                return Ok(SelectResult::InPlace);").unwrap();
        }
    }
}

fn emit_single_inst(
    output: &mut String,
    commits: &mut String,
    constructor: &Constructor,
    ctx: &InstEmitContext<'_>,
    request: InstEmitRequest,
) {
    let Constructor::Inst { opcode, args } = constructor else {
        writeln!(
            output,
            "                return Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Invalid constructor\")));"
        )
        .unwrap();
        return;
    };

    if let Some(inst_def) = ctx.final_inst_defs.get(opcode) {
        let implicit_uses: Vec<u32> = inst_def
            .implicit_uses
            .iter()
            .filter_map(|r| ctx.reg_map.get(r).copied())
            .collect();
        let implicit_defs: Vec<u32> = inst_def
            .implicit_defs
            .iter()
            .filter_map(|r| ctx.reg_map.get(r).copied())
            .collect();
        let ops_binding =
            if inst_def.operands.is_empty() && implicit_uses.is_empty() && implicit_defs.is_empty()
            {
                ""
            } else {
                "mut "
            };
        writeln!(
            output,
            "                #[allow(unused_mut)] let {}ops_{} = SmallVec::<[InstField; 4]>::new();",
            ops_binding, request.index
        )
        .unwrap();
        writeln!(output, "                #[allow(unused_mut)] let mut results_{} = SmallVec::<[Reg; 2]>::new();", request.index).unwrap();
        writeln!(
            output,
            "                #[allow(unused_mut)] let mut inputs_{} = SmallVec::<[Reg; 4]>::new();",
            request.index
        )
        .unwrap();
        let use_direct_schema_def = request.preserve_operands
            && ctx.schema_source_def_field.is_some()
            && inst_def_result_count(&inst_def.operands) == 1;
        if request.preserve_operands
            && inst_def_has_results(&inst_def.operands)
            && !use_direct_schema_def
        {
            writeln!(
                output,
                "                let source_defs_{index} = inst.results();",
                index = request.index
            )
            .unwrap();
            writeln!(
                output,
                "                let mut source_def_cursor_{index} = 0usize;",
                index = request.index
            )
            .unwrap();
        }

        let mut explicit_arg_cursor = 0usize;
        for (operand_index, operand) in inst_def.operands.iter().enumerate() {
            let use_source_def = request.preserve_operands
                && matches!(operand, OperandConstraint::Def(_))
                && args.len() - explicit_arg_cursor
                    == min_explicit_args_from(&inst_def.operands, operand_index);

            if use_source_def {
                if let Some(field) = ctx
                    .schema_source_def_field
                    .filter(|_| use_direct_schema_def)
                {
                    emit_schema_source_result(
                        output,
                        request.index,
                        operand,
                        ctx.schema_var
                            .expect("schema source defs require a schema binding"),
                        field,
                    );
                } else {
                    emit_source_result(output, request.index, operand);
                }
                continue;
            }

            let Some(arg) = args.get(explicit_arg_cursor) else {
                panic!(
                    "constructor {} is missing argument {} for target operand {}",
                    opcode, explicit_arg_cursor, operand_index
                );
            };
            emit_constructor_operand(
                output,
                request.index,
                arg,
                operand,
                ctx.var_map,
                ctx.reg_map,
            );
            explicit_arg_cursor += 1;
        }

        if explicit_arg_cursor != args.len() {
            panic!(
                "constructor {} has {} args but target schema consumed {}",
                opcode,
                args.len(),
                explicit_arg_cursor
            );
        }

        writeln!(
            commits,
            "                let inst_{} = store.writer().with_effects(&[{}], &[{}]).write(",
            request.index,
            implicit_uses
                .iter()
                .map(|r| format!("Reg::new_preg({r})"))
                .collect::<Vec<_>>()
                .join(", "),
            implicit_defs
                .iter()
                .map(|r| format!("Reg::new_preg({r})"))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        if ctx.final_inst_defs.contains_key(opcode) {
            writeln!(
                commits,
                "                    MachineOpcode::Target(TargetInst::{}.as_u32()),",
                opcode
            )
            .unwrap();
        } else {
            writeln!(
                commits,
                "                    MachineOpcode::Generic(GenericOpcode::{}),",
                opcode
            )
            .unwrap();
        }
        writeln!(commits, "                    &results_{},", request.index).unwrap();
        writeln!(
            commits,
            "                    &inputs_{}, &ops_{},",
            request.index, request.index
        )
        .unwrap();
        writeln!(commits, "                );").unwrap();
        writeln!(commits, "                out.push(inst_{});", request.index).unwrap();
    } else {
        panic!("constructor {opcode} has no instruction definition");
    }
}

fn inst_def_has_results(operands: &[OperandConstraint]) -> bool {
    operands
        .iter()
        .any(|op| matches!(op, OperandConstraint::Def(_)))
}

fn min_explicit_args_from(operands: &[OperandConstraint], start: usize) -> usize {
    operands[start..]
        .iter()
        .filter(|op| !matches!(op, OperandConstraint::Def(_)))
        .count()
}

fn inst_def_result_count(operands: &[OperandConstraint]) -> usize {
    operands
        .iter()
        .filter(|op| matches!(op, OperandConstraint::Def(_)))
        .count()
}

fn emit_source_result(output: &mut String, index: usize, operand: &OperandConstraint) {
    let _op_ctor = match operand {
        OperandConstraint::Def(_) => "Def",
        _ => panic!("source defs can only satisfy def-like operands"),
    };
    writeln!(
        output,
        r#"                {{
                    let reg = *source_defs_{index}
                        .get(source_def_cursor_{index})
                        .ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from("Source def mapping failed")))?;
                    source_def_cursor_{index} += 1;
                    results_{index}.push(reg);
                }}"#
    )
    .unwrap();
}

fn emit_schema_source_result(
    output: &mut String,
    index: usize,
    operand: &OperandConstraint,
    schema_var: &str,
    field: &str,
) {
    let _op_ctor = match operand {
        OperandConstraint::Def(_) => "Def",
        _ => panic!("schema source defs can only satisfy def-like operands"),
    };
    writeln!(
        output,
        r#"                results_{index}.push(reg_value({schema_var}.{field}.clone()).ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from("Schema reg mapping failed")))?);"#
    )
    .unwrap();
}

pub(crate) fn infer_schema_source_def_field(args: &[PatternArg]) -> Option<&str> {
    if named_args(args).any(|(field, _)| field == "dst") {
        Some("dst")
    } else {
        None
    }
}

fn emit_constructor_operand(
    output: &mut String,
    index: usize,
    arg: &Constructor,
    operand: &OperandConstraint,
    var_map: &HashMap<String, BindingSource>,
    reg_map: &HashMap<String, u32>,
) {
    let buffer = if matches!(operand, OperandConstraint::Def(_)) {
        "results"
    } else if matches!(
        operand,
        OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. }
    ) {
        "inputs"
    } else {
        "ops"
    };
    match arg {
        Constructor::Variable(name) => match var_map.get(name) {
            Some(BindingSource::SchemaValue) => {
                let push = schema_value_operand_expr(name, operand);
                writeln!(output, "                {buffer}_{index}.push({push});").unwrap();
            }
            Some(BindingSource::OperandIndex(op_index)) => {
                let push = operand_index_expr(*op_index, operand);
                writeln!(output, "                {buffer}_{index}.push({push});").unwrap();
            }
            None => {
                panic!("unknown constructor variable {}", name);
            }
        },
        Constructor::Imm(i) => match operand {
            OperandConstraint::Imm(_) => {
                writeln!(
                    output,
                    "                ops_{index}.push(InstField::Imm({i}));"
                )
                .unwrap();
            }
            _ => panic!("immediate constructor cannot satisfy non-immediate operand"),
        },
        Constructor::Reg(name) => {
            let enc = reg_map
                .get(name)
                .unwrap_or_else(|| panic!("unknown physical register {}", name));
            let push = match operand {
                OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => {
                    format!("Reg::new_preg({enc})")
                }
                OperandConstraint::Def(_) => {
                    format!("Reg::new_preg({enc})")
                }
                OperandConstraint::StackSlot(_) => {
                    panic!("physical register constructor cannot satisfy a stackslot operand")
                }
                _ => panic!("physical register constructor cannot satisfy this operand kind"),
            };
            writeln!(output, "                {buffer}_{index}.push({push});").unwrap();
        }
        Constructor::Inst { .. } => {
            panic!("nested constructor is not a valid target operand");
        }
    }
}

fn schema_value_operand_expr(name: &str, operand: &OperandConstraint) -> String {
    let rust_name = rust_ident(name);
    match operand {
        OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => format!(
            "reg_value({rust_name}).ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Schema reg mapping failed\")))?"
        ),
        OperandConstraint::Def(_) => format!(
            "reg_value({rust_name}).ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Schema reg mapping failed\")))?"
        ),
        OperandConstraint::Imm(_) => format!("InstField::Imm({rust_name}.into())"),
        OperandConstraint::Block(_) => format!("InstField::Block({rust_name})"),
        OperandConstraint::Global(_) => format!("InstField::Global({rust_name})"),
        OperandConstraint::StackSlot(_) => format!("InstField::StackSlot({rust_name})"),
    }
}

fn rust_ident(name: &str) -> String {
    match name {
        "as" | "break" | "const" | "continue" | "crate" | "else" | "enum" | "extern" | "false"
        | "fn" | "for" | "if" | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move"
        | "mut" | "pub" | "ref" | "return" | "self" | "Self" | "static" | "struct" | "super"
        | "trait" | "true" | "type" | "unsafe" | "use" | "where" | "while" | "async" | "await"
        | "dyn" | "abstract" | "become" | "box" | "do" | "final" | "macro" | "override"
        | "priv" | "try" | "typeof" | "unsized" | "virtual" | "yield" => format!("r#{}", name),
        _ => name.to_string(),
    }
}

fn operand_index_expr(op_index: usize, operand: &OperandConstraint) -> String {
    match operand {
        OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => format!(
            "inst.inputs().get({op_index}).copied().ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Operand reg mapping failed\")))?"
        ),
        OperandConstraint::Def(_) => format!(
            "inst.inputs().get({op_index}).copied().ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Operand reg mapping failed\")))?"
        ),
        OperandConstraint::Imm(_) => format!(
            "match field_by_index(inst, {op_index}) {{ Some(InstField::Imm(v)) => InstField::Imm(v), _ => return Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Operand immediate mapping failed\"))), }}"
        ),
        OperandConstraint::Block(_) => format!(
            "match field_by_index(inst, {op_index}) {{ Some(InstField::Block(v)) => InstField::Block(v), _ => return Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Operand block mapping failed\"))), }}"
        ),
        OperandConstraint::Global(_) => format!(
            "match field_by_index(inst, {op_index}) {{ Some(InstField::Global(v)) => InstField::Global(v), _ => return Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Operand global mapping failed\"))), }}"
        ),
        OperandConstraint::StackSlot(_) => format!(
            "match field_by_index(inst, {op_index}) {{ Some(InstField::StackSlot(v)) => InstField::StackSlot(v), _ => return Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"Operand stackslot mapping failed\"))), }}"
        ),
    }
}

pub(crate) fn generate_generic_inst_metadata(
    output: &mut String,
    module: &crate::target::ast::Module,
    final_inst_defs: &HashMap<String, FinalInstDef>,
) {
    let metadata_map = derive_generic_inst_metadata(module, final_inst_defs);

    if metadata_map.is_empty() {
        writeln!(
            output,
            "\npub fn generic_inst_metadata(_opcode: veloc_lir::GenericOpcode) -> &'static GenericInstMetadata {{\n    &GenericInstMetadata::EMPTY\n}}"
        )
        .unwrap();
        return;
    }

    writeln!(
        output,
        "\n/// Generic instruction metadata inferred from select rules."
    )
    .unwrap();
    for (opcode, metadata) in &metadata_map {
        let const_name = format!(
            "GENERIC_INST_{}_METADATA",
            sanitize_ident(opcode).to_ascii_uppercase()
        );
        let fixed_entries = metadata
            .fixed_uses
            .iter()
            .map(|(use_operand, reg)| {
                format!(
                    "FixedUseConstraint {{ use_operand: {}, reg: {} }}",
                    use_operand,
                    reg_const_name(reg)
                )
            })
            .collect();

        writeln!(
            output,
            "pub const {const_name}: GenericInstMetadata = GenericInstMetadata {{"
        )
        .unwrap();
        writeln!(output, "    fixed_uses: {},", format_slice(fixed_entries)).unwrap();
        writeln!(output, "}};").unwrap();
    }

    writeln!(
        output,
        "\npub fn generic_inst_metadata(opcode: veloc_lir::GenericOpcode) -> &'static GenericInstMetadata {{"
    )
    .unwrap();
    writeln!(output, "    match opcode {{").unwrap();
    for opcode in metadata_map.keys() {
        let const_name = format!(
            "GENERIC_INST_{}_METADATA",
            sanitize_ident(opcode).to_ascii_uppercase()
        );
        writeln!(
            output,
            "        veloc_lir::GenericOpcode::{opcode} => &{const_name},",
            opcode = opcode,
            const_name = const_name
        )
        .unwrap();
    }
    writeln!(output, "        _ => &GenericInstMetadata::EMPTY,").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DerivedGenericInstMetadata {
    fixed_uses: Vec<(usize, String)>,
}

fn schema_field_operand_index(schema: &str, field: &str) -> Option<usize> {
    match (schema, field) {
        ("BinaryReg", "lhs") => Some(0),
        ("BinaryReg", "rhs") => Some(1),
        _ => None,
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

fn derive_generic_inst_metadata(
    module: &crate::target::ast::Module,
    final_inst_defs: &HashMap<String, FinalInstDef>,
) -> BTreeMap<String, DerivedGenericInstMetadata> {
    let mut result = BTreeMap::<String, DerivedGenericInstMetadata>::new();

    for def in &module.defs {
        let Def::SelectRule(rule) = def else {
            continue;
        };
        let Some(pattern) = rule.patterns.first() else {
            continue;
        };
        let Some(schema_name) = pattern_schema_name(pattern) else {
            continue;
        };
        if schema_name != "BinaryReg" {
            continue;
        }
        let Some(opcode) = pattern_opcode(pattern) else {
            continue;
        };
        let Some(pattern_args) = pattern_args(pattern) else {
            continue;
        };
        let Constructor::Inst {
            opcode: target_opcode,
            args: constructor_args,
        } = &rule.emit
        else {
            continue;
        };
        let Some(target_inst_def) = final_inst_defs.get(target_opcode) else {
            continue;
        };

        let field_bindings = collect_field_variable_bindings(pattern_args);
        let schema_source_def_field = infer_schema_source_def_field(pattern_args);
        let target_arg_bindings = constructor_arg_bindings_by_target_operand(
            &target_inst_def.operands,
            constructor_args,
            schema_source_def_field,
        );
        let metadata = result.entry(opcode.to_string()).or_default();

        for (target_operand_index, operand) in target_inst_def.operands.iter().enumerate() {
            let OperandConstraint::FixedUse { reg, .. } = operand else {
                continue;
            };
            let Some(Some(Constructor::Variable(var_name))) =
                target_arg_bindings.get(target_operand_index)
            else {
                continue;
            };
            let Some(field) = field_bindings.get(var_name) else {
                continue;
            };
            let Some(source_operand_index) = schema_field_operand_index(schema_name, field) else {
                continue;
            };
            let fixed = (source_operand_index, reg.clone());
            if !metadata.fixed_uses.contains(&fixed) {
                metadata.fixed_uses.push(fixed);
            }
        }
    }

    for metadata in result.values_mut() {
        metadata.fixed_uses.sort_unstable();
        metadata.fixed_uses.dedup();
    }

    result.retain(|_, metadata| !metadata.fixed_uses.is_empty());

    result
}

pub(super) fn check_temps(rule: &crate::target::ast::SelectRuleDef) -> Result<(), String> {
    let mut bindings = collect_var_bindings(&rule.patterns[0]);
    for (name, like) in &rule.temps {
        if !bindings.contains_key(like) || bindings.contains_key(name) {
            return Err(format!(
                "invalid temporary ${name}: exemplar ${like} must be bound and name must be fresh"
            ));
        }
        bindings.insert(name.clone(), BindingSource::SchemaValue);
    }
    Ok(())
}

fn emit_temps(
    output: &mut String,
    temps: &[(String, String)],
    bindings: &mut HashMap<String, BindingSource>,
) {
    for (name, like) in temps {
        let exemplar = match bindings[like] {
            BindingSource::SchemaValue => format!("reg_value({})", rust_ident(like)),
            BindingSource::OperandIndex(index) => {
                format!("inst.inputs().get({index}).copied()")
            }
        };
        writeln!(
            output,
            "let {} = ctx.alloc_tmp({exemplar}.expect(\"temporary exemplar must be a register\"));",
            rust_ident(name)
        )
        .unwrap();
        bindings.insert(name.clone(), BindingSource::SchemaValue);
    }
}

pub(crate) fn generate_select_instruction(
    output: &mut String,
    module: &crate::target::ast::Module,
    extractors: &HashMap<String, ExtractorDef>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    arch: &str,
) {
    let reg_map = collect_reg_ids(module);
    let needs_positional_helpers = module_has_positional_rules(module);
    let decls = collect_decl_map(module);
    let opcode_rules = collect_select_rules_by_opcode(module);
    let used_extractors = collect_used_extractors(module, extractors);

    let extra_bound = if arch == "x86_64" {
        " + crate::target::x86_64::lowering::X86LoweringContext"
    } else {
        ""
    };

    writeln!(
        output,
        r#"
pub fn select_instructions<C: LoweringContext{extra_bound}>(
    ctx: &mut C,
    store: &mut veloc_lir::InstStore,
    source: veloc_lir::InstId,
    out: &mut alloc::vec::Vec<veloc_lir::InstId>,
) -> Result<SelectResult, crate::error::Error> {{
    use veloc_lir::{{GenericOpcode, MachineOpcode, VReg}};
    let inst = &store.get(source);
    let decoded = &veloc_lir::InstRead::view(*inst);
    use crate::target::arch::SelectResult;
    use crate::target::{arch}::isle::TargetInst;

"#,
        extra_bound = extra_bound,
        arch = arch,
    )
    .unwrap();
    if needs_positional_helpers {
        writeln!(output, "    let v0 = vreg_by_index(inst, 0);").unwrap();
        writeln!(output, "    let v1 = vreg_by_index(inst, 1);").unwrap();
        writeln!(output, "    let v2 = vreg_by_index(inst, 2);").unwrap();
    }
    writeln!(output).unwrap();

    for name in used_extractors {
        let exc = extractors
            .get(&name)
            .unwrap_or_else(|| panic!("missing extractor definition for {}", name));
        let cond = generate_pattern_condition(&exc.body, "v", &decls);
        let value_param = if pattern_condition_needs_value(&exc.body, &decls) {
            "v"
        } else {
            "_v"
        };
        writeln!(
            output,
            "    let is_{} = |v_opt: Option<VReg>| v_opt.map_or(false, |{}| {});",
            name.to_lowercase(),
            value_param,
            cond
        )
        .unwrap();
    }
    writeln!(output).unwrap();

    writeln!(
        output,
        "    let MachineOpcode::Generic(opcode) = &inst.opcode() else {{"
    )
    .unwrap();
    writeln!(output, "        return Ok(SelectResult::Keep);").unwrap();
    writeln!(output, "    }};").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "    // 尝试按规则选择指令序列").unwrap();
    writeln!(output, "    match opcode {{").unwrap();

    let mut opcodes: Vec<_> = opcode_rules.keys().cloned().collect();
    opcodes.sort();

    for op in opcodes {
        writeln!(output, "        GenericOpcode::{} => {{", op).unwrap();
        let rules = &opcode_rules[&op];
        let mut rule_index = 0usize;
        while rule_index < rules.len() {
            let rule = rules[rule_index];
            let Some(pattern) = rule.patterns.first() else {
                rule_index += 1;
                continue;
            };
            let Some(args) = pattern_args(pattern) else {
                rule_index += 1;
                continue;
            };

            if let Some(schema_name) = pattern_schema_name(pattern) {
                let mut group_end = rule_index;
                while group_end < rules.len() {
                    let Some(group_pattern) = rules[group_end].patterns.first() else {
                        group_end += 1;
                        continue;
                    };
                    if pattern_schema_name(group_pattern) != Some(schema_name) {
                        break;
                    }
                    group_end += 1;
                }

                let schema_var =
                    if schema_group_needs_binding(&rules[rule_index..group_end], extractors) {
                        "schema_inst"
                    } else {
                        "_schema_inst"
                    };
                writeln!(
                    output,
                    "            if let veloc_lir::InstView::{}({}) = decoded {{",
                    schema_name, schema_var
                )
                .unwrap();
                emit_schema_value_bindings_for_group(
                    output,
                    &rules[rule_index..group_end],
                    schema_var,
                );

                while rule_index < group_end {
                    let grouped_rule = rules[rule_index];
                    let Some(grouped_pattern) = grouped_rule.patterns.first() else {
                        rule_index += 1;
                        continue;
                    };
                    let Some(grouped_args) = pattern_args(grouped_pattern) else {
                        rule_index += 1;
                        continue;
                    };
                    let mut var_map = collect_var_bindings(grouped_pattern);
                    let conditions = collect_schema_rule_conditions(
                        grouped_args,
                        extractors,
                        schema_var,
                        schema_name,
                    );
                    if conditions.is_empty() {
                        writeln!(output, "                {{").unwrap();
                    } else {
                        writeln!(output, "                if {} {{", conditions.join(" && "))
                            .unwrap();
                    }
                    emit_temps(output, &grouped_rule.temps, &mut var_map);
                    let emit_ctx = InstEmitContext {
                        var_map: &var_map,
                        final_inst_defs,
                        reg_map: &reg_map,
                        schema_var: Some(schema_var),
                        schema_source_def_field: infer_schema_source_def_field(grouped_args),
                    };
                    emit_constructor_sequence(
                        output,
                        &grouped_rule.emit,
                        &emit_ctx,
                        InstEmitRequest {
                            index: 0,
                            preserve_operands: true,
                        },
                    );
                    writeln!(output, "                }}").unwrap();
                    rule_index += 1;
                }

                writeln!(output, "            }}").unwrap();
            } else {
                let mut var_map = collect_var_bindings(pattern);
                let conditions = collect_positional_rule_conditions(args, extractors);

                if conditions.is_empty() {
                    writeln!(output, "            {{").unwrap();
                } else {
                    writeln!(output, "            if {} {{", conditions.join(" && ")).unwrap();
                }

                emit_temps(output, &rule.temps, &mut var_map);
                let emit_ctx = InstEmitContext {
                    var_map: &var_map,
                    final_inst_defs,
                    reg_map: &reg_map,
                    schema_var: None,
                    schema_source_def_field: None,
                };
                emit_constructor_sequence(
                    output,
                    &rule.emit,
                    &emit_ctx,
                    InstEmitRequest {
                        index: 0,
                        preserve_operands: true,
                    },
                );
                writeln!(output, "            }}").unwrap();
                rule_index += 1;
            }
        }
        writeln!(output, "        }}").unwrap();
    }
    writeln!(output, "        _ => {{}}").unwrap();
    writeln!(output, "    }}").unwrap();

    writeln!(
        output,
        "    Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"No matching ISLE rule found for instruction\")))"
    )
    .unwrap();
    writeln!(output, "}}").unwrap();
}
