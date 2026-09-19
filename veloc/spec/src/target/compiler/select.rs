use crate::target::ast::{CondCode, Constructor, DeclDef, Def, Pattern, SelectRuleDef};
use crate::target::{ExtractorDef, OperandConstraint, PatternArg};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

use super::FinalInstDef;
use super::generate::{collect_reg_ids, format_slice, reg_const_name, sanitize_ident};

fn strip_node_binds(pattern: &Pattern) -> &Pattern {
    pattern.strip_node_binds()
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum BindingSource {
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

fn pattern_condition_needs_value(pattern: &Pattern, decls: &HashMap<String, DeclDef>) -> bool {
    match strip_node_binds(pattern) {
        Pattern::And(parts) => parts
            .iter()
            .any(|part| pattern_condition_needs_value(part, decls)),
        Pattern::Typed { .. } => true,
        Pattern::Schema { .. }
        | Pattern::Variable(_)
        | Pattern::IntConst(_)
        | Pattern::CondCode(_)
        | Pattern::StackSlot(_)
        | Pattern::Block(_) => false,
        Pattern::Opcode { opcode, args, .. } => match opcode.as_str() {
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
        Pattern::Typed { .. }
        | Pattern::Variable(_)
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
        for (_, pattern) in named_args(&rule.fields) {
            collect_used_extractors_in_pattern(pattern, extractors, &mut used);
        }
    }
    used
}

fn collect_schema_rule_conditions(
    rule: &SelectRuleDef,
    extractors: &HashMap<String, ExtractorDef>,
    schema_var: &str,
) -> Vec<String> {
    named_args(&rule.fields)
        .filter_map(|(field, pattern)| {
            let schema_name = field
                .split_once('.')
                .and_then(|(owner, _)| {
                    rule.definitions
                        .iter()
                        .find(|p| p.name == owner)
                        .map(|p| p.schema.as_str())
                })
                .unwrap_or(&rule.schema);
            collect_schema_field_conditions(field, pattern, extractors, schema_var, schema_name)
                .map(|parts| parts.join(" && "))
        })
        .filter(|cond| !cond.is_empty())
        .collect()
}

fn field_expr(field: &str, root: &str) -> String {
    if let Some((producer, member)) = field.split_once('.') {
        format!("def_{producer}.{member}")
    } else {
        format!("{root}.{field}")
    }
}

fn collect_schema_field_conditions(
    field: &str,
    pattern: &Pattern,
    extractors: &HashMap<String, ExtractorDef>,
    schema_var: &str,
    schema_name: &str,
) -> Option<Vec<String>> {
    let value = field_expr(field, schema_var);
    match strip_node_binds(pattern) {
        Pattern::Typed { types, .. } => Some(vec![format!(
            "reg_value_to_vreg({value}).is_some_and(|reg| [{}].contains(&ctx.get_type(reg)))",
            types.join(", ")
        )]),
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
        Pattern::IntConst(value) => Some(vec![format!(
            "{} == {}",
            field_expr(field, schema_var),
            value
        )]),
        Pattern::CondCode(cc) => {
            render_cond_code_match(schema_name, *cc).map(|expr| vec![format!("{value} == {expr}")])
        }
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

fn collect_var_bindings(args: &[PatternArg]) -> HashMap<String, BindingSource> {
    let mut map = HashMap::new();
    for (_, arg) in named_args(args) {
        collect_vars_in_pattern(arg, BindingSource::SchemaValue, &mut map);
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
        for build in &rule.builds {
            collect_constructor_variables(build, &mut needed_vars);
        }
        needed_vars.extend(rule.temps.iter().map(|(_, like)| like.clone()));
    }

    let mut bindings = BTreeMap::new();
    for rule in rules {
        let args = &rule.fields;
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
            "                let {} = {}.clone();",
            rust_var,
            field_expr(&field, schema_var)
        )
        .unwrap();
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
        for build in &rule.builds {
            collect_constructor_variables(build, &mut needed_vars);
        }
    }

    for rule in rules {
        let args = &rule.fields;
        if !rule.definitions.is_empty() {
            return true;
        }

        if infer_schema_source_def_field(args).is_some() {
            return true;
        }
        if !collect_schema_rule_conditions(rule, extractors, "schema_inst").is_empty() {
            return true;
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
        Pattern::Variable(name) | Pattern::Typed { name, .. } => {
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
    constructors: &[Constructor],
    ctx: &InstEmitContext<'_>,
) {
    // Preparation decodes operands without writing instructions.
    let mut commits = String::new();
    let single = constructors.len() == 1;
    for (index, ctor) in constructors.iter().enumerate() {
        emit_single_inst(
            output,
            &mut commits,
            ctor,
            ctx,
            InstEmitRequest {
                index,
                preserve_operands: single,
            },
        );
    }
    output.push_str(&commits);
    let result = if single { "InPlace" } else { "Replace" };
    writeln!(output, "                return Ok(SelectResult::{result});").unwrap();
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
        let ops_binding = if inst_def.operands.is_empty() {
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
        }

        let mut explicit_arg_cursor = 0usize;
        let mut source_def_cursor = 0usize;
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
                    emit_source_result(output, request.index, source_def_cursor);
                    source_def_cursor += 1;
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
            "                let inst_{} = TargetInst::{opcode}.write(store.writer(),",
            request.index
        )
        .unwrap();
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

fn emit_source_result(output: &mut String, index: usize, source_def: usize) {
    writeln!(
        output,
        r#"                {{
                    let reg = *source_defs_{index}
                        .get({source_def})
                        .ok_or_else(|| crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from("Source def mapping failed")))?;
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
        if rule.schema != "BinaryReg" {
            continue;
        }
        let opcode = &rule.opcode;
        let pattern_args = &rule.fields;
        let [
            Constructor::Inst {
                opcode: target_opcode,
                args: constructor_args,
            },
        ] = rule.builds.as_slice()
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
            let Some(source_operand_index) = schema_field_operand_index(&rule.schema, field) else {
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

pub(super) fn check_construction(
    rule: &SelectRuleDef,
    instructions: &HashMap<String, FinalInstDef>,
) -> Result<(), String> {
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
    }
    Ok(())
}

pub(super) fn check_temps(rule: &crate::target::ast::SelectRuleDef) -> Result<(), String> {
    let mut bindings = collect_var_bindings(&rule.fields);
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
    context: &str,
) {
    let reg_map = collect_reg_ids(module);
    let decls = collect_decl_map(module);
    let opcode_rules = collect_select_rules_by_opcode(module);
    let used_extractors = collect_used_extractors(module, extractors);

    writeln!(
        output,
        r#"
pub fn select_instructions<C: LoweringContext + crate::target::arch::TargetFeatures<Features = FeatureSet> + {context}>(
    ctx: &mut C,
    store: &mut veloc_lir::InstBuilder<'_>,
    source: veloc_lir::InstId,
    out: &mut alloc::vec::Vec<veloc_lir::InstId>,
) -> Result<SelectResult, crate::error::Error> {{
    use veloc_lir::{{GenericOpcode, MachineOpcode}};
    let inst = &store.get(source);
    let decoded = &veloc_lir::InstRead::view(*inst);
    use crate::target::arch::SelectResult;
    use crate::target::{arch}::inst::TargetInst;

"#,
        context = context,
        arch = arch,
    )
    .unwrap();
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
            "    let is_{} = |v_opt: Option<veloc_lir::VReg>| v_opt.map_or(false, |{}| {});",
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
        let schema = &rules[0].schema;
        let schema_var = if schema_group_needs_binding(rules, extractors) {
            "schema_inst"
        } else {
            "_schema_inst"
        };
        writeln!(
            output,
            "            if let veloc_lir::InstView::{schema}({schema_var}) = decoded {{"
        )
        .unwrap();
        // Checks are pure: identical type/feature predicates can be evaluated once
        // without moving allocations or construction out of their selected case.
        let candidate_conditions = rules
            .iter()
            .map(|rule| {
                let mut conditions = collect_schema_rule_conditions(rule, extractors, schema_var);
                for build in &rule.builds {
                    feature_conditions(build, final_inst_defs, &mut conditions);
                }
                conditions
            })
            .collect::<Vec<_>>();
        let mut counts = BTreeMap::<&str, usize>::new();
        for conditions in &candidate_conditions {
            for condition in conditions {
                *counts.entry(condition).or_default() += 1;
            }
        }
        let shared = counts
            .into_iter()
            .filter(|(condition, count)| *count > 1 && !condition.contains("def_"))
            .enumerate()
            .map(|(id, (condition, _))| (condition, format!("check_{id}")))
            .collect::<BTreeMap<_, _>>();
        for (condition, name) in &shared {
            writeln!(output, "                let {name} = {condition};").unwrap();
        }
        for (rule, conditions) in rules.iter().zip(&candidate_conditions) {
            // Every lookup and opcode test falls through to the next candidate.
            // Definition lookup is independent of the fold-safety check below.
            for p in &rule.definitions {
                let input = field_expr(&p.input, schema_var);
                writeln!(output, "if let Some(def_{0}_id) = reg_value({input}).and_then(|reg| store.def(reg)) {{", p.name).unwrap();
                writeln!(output, "let def_{0}_inst = store.get(def_{0}_id);", p.name).unwrap();
                writeln!(
                    output,
                    "if def_{0}_inst.generic_opcode() == Some(GenericOpcode::{1}) {{",
                    p.name, p.opcode
                )
                .unwrap();
                writeln!(output, "if let veloc_lir::InstView::{1}(def_{0}) = veloc_lir::InstRead::view(def_{0}_inst) {{", p.name, p.schema).unwrap();
            }
            let mut var_map = collect_var_bindings(&rule.fields);
            let conditions = conditions
                .iter()
                .map(|condition| {
                    shared
                        .get(condition.as_str())
                        .map(String::as_str)
                        .unwrap_or(condition.as_str())
                })
                .collect::<Vec<_>>();
            if conditions.is_empty() {
                writeln!(output, "                {{").unwrap();
            } else {
                writeln!(output, "                if {} {{", conditions.join(" && ")).unwrap();
            }
            // Until effect/dependency legality is modeled, committing a graph
            // rewrite requires pure, single-result matched definitions.
            if !rule.definitions.is_empty() {
                let safe = rule
                    .definitions
                    .iter()
                    .map(|p| {
                        format!(
                            "def_{0}_inst.is_pure_value() && def_{0}_inst.results().len() == 1",
                            p.name
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" && ");
                writeln!(output, "if {safe} {{").unwrap();
            }
            emit_schema_value_bindings_for_group(output, &[*rule], schema_var);
            emit_temps(output, &rule.temps, &mut var_map);
            let emit_ctx = InstEmitContext {
                var_map: &var_map,
                final_inst_defs,
                reg_map: &reg_map,
                schema_var: Some(schema_var),
                schema_source_def_field: infer_schema_source_def_field(&rule.fields),
            };
            emit_constructor_sequence(output, &rule.builds, &emit_ctx);
            if !rule.definitions.is_empty() {
                writeln!(output, "}}").unwrap();
            }
            writeln!(output, "                }}").unwrap();
            for _ in &rule.definitions {
                writeln!(output, "}} }} }}").unwrap();
            }
        }
        writeln!(output, "            }}").unwrap();
        writeln!(output, "        }}").unwrap();
    }
    writeln!(output, "        _ => {{}}").unwrap();
    writeln!(output, "    }}").unwrap();

    writeln!(
        output,
        "    Err(crate::error::Error::select(inst.opcode().clone(), alloc::string::String::from(\"No matching selection rule found for instruction\")))"
    )
    .unwrap();
    writeln!(output, "}}").unwrap();
}

fn feature_conditions(
    ctor: &Constructor,
    instructions: &HashMap<String, FinalInstDef>,
    out: &mut Vec<String>,
) {
    if let Constructor::Inst { opcode, args } = ctor {
        if instructions
            .get(opcode)
            .is_some_and(|inst| !inst.requires.is_empty())
        {
            out.push(format!(
                "ctx.supports_features(TargetInst::{opcode}.required_features())"
            ));
        }
        for arg in args {
            feature_conditions(arg, instructions, out);
        }
    }
}
