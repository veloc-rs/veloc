use crate::ast::{CondCode, Constructor, DeclDef, Def, Pattern, RuleAttrs, SelectRuleDef};
use crate::{parser, EmitExpr, Expr, ExtractorDef, MacroDef, OperandConstraint, PatternArg};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

// =============================================================================
// 表达式生成 - 辅助函数
// =============================================================================

/// 查找操作数索引及约束类型
fn find_operand_info<'a>(
    var_name: &str,
    operands: &'a [OperandConstraint],
) -> Option<(usize, &'a OperandConstraint)> {
    operands.iter().enumerate().find(|(_, op)| match op {
        OperandConstraint::Use(name)
        | OperandConstraint::FixedUse { src: name, .. }
        | OperandConstraint::Def(name)
        | OperandConstraint::Imm(name)
        | OperandConstraint::Block(name)
        | OperandConstraint::Global(name)
        | OperandConstraint::StackSlot(name) => name == var_name,
        OperandConstraint::TiedDef { dst, src } => dst == var_name || src == var_name,
    })
}

/// 生成变量引用代码
fn generate_variable(var_name: &str, operands: &[OperandConstraint]) -> String {
    match find_operand_info(var_name, operands) {
        Some((index, constraint)) => {
            let arm = match constraint {
                OperandConstraint::Imm(_) => "MachineOperand::Imm(val) => *val as u64",
                OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => {
                    "MachineOperand::Use(reg) => reg.index() as u64"
                }
                OperandConstraint::Def(_) => {
                    "MachineOperand::Def(reg) => reg.to_reg().index() as u64"
                }
                OperandConstraint::TiedDef { .. } => {
                    "MachineOperand::TiedDefUse(reg) => reg.to_reg().index() as u64"
                }
                OperandConstraint::Block(_)
                | OperandConstraint::Global(_)
                | OperandConstraint::StackSlot(_) => {
                    return format!("/* non-numeric operand {} */ 0", var_name);
                }
            };
            format!("(match &inst.operands[{}] {{ {}, _ => return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!(\"Operand type mismatch at index {} for {{}}\", \"{}\"))) }})", index, arm, index, var_name)
        }
        None => "0".to_string(),
    }
}

/// 生成寄存器编码代码
fn generate_hw_enc(var_name: &str, operands: &[OperandConstraint]) -> String {
    match find_operand_info(var_name, operands) {
        Some((index, constraint)) => {
            let arm = match constraint {
                OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => {
                    "MachineOperand::Use(reg) => reg.index() as u8"
                }
                OperandConstraint::Def(_) => {
                    "MachineOperand::Def(reg) => reg.to_reg().index() as u8"
                }
                OperandConstraint::TiedDef { .. } => {
                    "MachineOperand::TiedDefUse(reg) => reg.to_reg().index() as u8"
                }
                OperandConstraint::Imm(_) => "MachineOperand::Imm(val) => *val as u8", // 硬件编码中偶尔会用到立即数作为掩码/扩展
                OperandConstraint::Block(_)
                | OperandConstraint::Global(_)
                | OperandConstraint::StackSlot(_) => {
                    return format!(
                        "/* hw-enc {} not available for non-reg operand */ 0",
                        var_name
                    );
                }
            };
            format!(
                "(match &inst.operands[{}] {{ {}, _ => return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!(\"Operand type mismatch at index {} for {{}}\", \"{}\"))) }} as u64)",
                index, arm, index, var_name
            )
        }
        None => format!("/* hw-enc {} not found */ 0", var_name),
    }
}

fn generate_stack_slot_expr(var_name: &str, operands: &[OperandConstraint], field: &str) -> String {
    match find_operand_info(var_name, operands) {
        Some((index, OperandConstraint::StackSlot(_))) => {
            let access = match field {
                "base_hw_enc" => "mfunc.stack_frame.slots[slot].base_reg.index() as u64",
                "offset" => "mfunc.stack_frame.slots[slot].offset as i64",
                "size" => "mfunc.stack_frame.slots[slot].size as i64",
                "align" => "mfunc.stack_frame.slots[slot].align as i64",
                other => panic!("unknown stack slot field {}", other),
            };
            format!(
                r#"{{
                    let slot = match &inst.operands[{index}] {{
                        MachineOperand::StackSlot(slot) => *slot,
                        _ => return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!("Operand type mismatch at index {index} for {{}}", "{var_name}"))),
                    }};
                    {access}
                }}"#
            )
        }
        Some((index, _)) => format!(
            "/* operand {} at index {} is not a stackslot */ 0",
            var_name, index
        ),
        None => format!("/* stackslot {} not found */ 0", var_name),
    }
}

/// 生成表达式代码
fn generate_expr(
    expr: &Expr,
    operands: &[OperandConstraint],
    macros: &HashMap<String, MacroDef>,
) -> String {
    match expr {
        Expr::Int(i) => i.to_string(),
        Expr::Variable(v) => generate_variable(v, operands),
        Expr::HwEnc(v) => generate_hw_enc(v, operands),
        Expr::SlotBaseHwEnc(v) => generate_stack_slot_expr(v, operands, "base_hw_enc"),
        Expr::SlotOffset(v) => generate_stack_slot_expr(v, operands, "offset"),
        Expr::SlotSize(v) => generate_stack_slot_expr(v, operands, "size"),
        Expr::SlotAlign(v) => generate_stack_slot_expr(v, operands, "align"),
        Expr::BitOr(a, b) => format!(
            "({} | {})",
            generate_expr(a, operands, macros),
            generate_expr(b, operands, macros)
        ),
        Expr::BitAnd(a, b) => format!(
            "({} & {})",
            generate_expr(a, operands, macros),
            generate_expr(b, operands, macros)
        ),
        Expr::Shl(a, b) => format!(
            "({} << {})",
            generate_expr(a, operands, macros),
            generate_expr(b, operands, macros)
        ),
        Expr::Shr(a, b) => format!(
            "({} >> {})",
            generate_expr(a, operands, macros),
            generate_expr(b, operands, macros)
        ),
        Expr::Call(name, args) => {
            // 尝试展开宏
            if let Some(m) = macros.get(name) {
                let args_map: HashMap<_, _> = m
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| args.get(i).map(|val| (name.clone(), val.clone())))
                    .collect();
                let expanded = subst_expr(&m.body, &args_map);
                return generate_expr(&expanded, operands, macros);
            }

            let gen_args: Vec<String> = args
                .iter()
                .map(|a| generate_expr(a, operands, macros))
                .collect();

            match (name.as_str(), gen_args.len()) {
                ("bit-or", 2) => format!("({} | {})", gen_args[0], gen_args[1]),
                ("bit-and", 2) => format!("({} & {})", gen_args[0], gen_args[1]),
                ("shl", 2) => format!("({} << {})", gen_args[0], gen_args[1]),
                ("shr", 2) => format!("({} >> {})", gen_args[0], gen_args[1]),
                _ => format!("ctx.{}({})", name.replace("-", "_"), gen_args.join(", ")),
            }
        }
    }
}

/// 生成 emit 表达式代码
fn generate_emit_expr(
    expr: &EmitExpr,
    operands: &[OperandConstraint],
    macros: &HashMap<String, MacroDef>,
) -> String {
    match expr {
        EmitExpr::Byte(b) => format!("emitter.write_bytes(&[{:#04x}]);", b),
        EmitExpr::ByteExpr(e) => {
            let val = generate_expr(e, operands, macros);
            format!("emitter.write_bytes(&[({}) as u8]);", val)
        }
        EmitExpr::Imm16(e) => {
            let val = generate_expr(e, operands, macros);
            format!("emitter.write_bytes(&(({} as u16).to_le_bytes()));", val)
        }
        EmitExpr::Imm32(e) => {
            let val = generate_expr(e, operands, macros);
            format!("emitter.write_bytes(&(({} as u32).to_le_bytes()));", val)
        }
        EmitExpr::Imm64(e) => {
            let val = generate_expr(e, operands, macros);
            format!("emitter.write_bytes(&(({} as u64).to_le_bytes()));", val)
        }
        EmitExpr::Rel32(name) => match find_operand_info(name, operands) {
            Some((index, OperandConstraint::Block(_))) => format!(
                r#"{{
                    let disp_offset = emitter.position();
                    emitter.write_bytes(&[0, 0, 0, 0]);
                    let next_offset = emitter.position();
                    match &inst.operands[{index}] {{
                        MachineOperand::Block(target) => {{
                            emitter.add_block_rel32_fixup(disp_offset, next_offset, *target);
                        }}
                        _ => return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!("Operand type mismatch at index {index} for {{}}", "{name}"))),
                    }}
                }}"#
            ),
            Some((index, OperandConstraint::Global(_))) => format!(
                r#"{{
                    let disp_offset = emitter.position();
                    emitter.write_bytes(&[0, 0, 0, 0]);
                    let next_offset = emitter.position();
                    match &inst.operands[{index}] {{
                        MachineOperand::Global(target) => {{
                            emitter.add_global_rel32_fixup(disp_offset, next_offset, target.clone());
                        }}
                        _ => return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!("Operand type mismatch at index {index} for {{}}", "{name}"))),
                    }}
                }}"#
            ),
            Some((index, _)) => format!(
                r#"return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!("rel32 operand at index {} for {{}} must be a block or global target", "{}")));"#,
                index, name
            ),
            None => format!(
                r#"return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!("rel32 operand {{}} not found", "{}")));"#,
                name
            ),
        },
        EmitExpr::If(cond, then_p, else_p) => {
            let mut s = format!("if ({}) != 0 {{\n", generate_expr(cond, operands, macros));
            for e in then_p {
                s.push_str(&format!(
                    "                {}\n",
                    generate_emit_expr(e, operands, macros)
                ));
            }
            s.push_str("            } else {\n");
            for e in else_p {
                s.push_str(&format!(
                    "                {}\n",
                    generate_emit_expr(e, operands, macros)
                ));
            }
            s.push_str("            }");
            s
        }
    }
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
                .collect();
            format!("({})", conds.join(" && "))
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
                .map(|arg| format!("!{}", generate_pattern_condition(arg, var_name, decls)))
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

fn module_has_positional_rules(module: &crate::ast::Module) -> bool {
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

fn module_needs_source_defs_helper(
    module: &crate::ast::Module,
    final_inst_defs: &HashMap<String, FinalInstDef>,
) -> bool {
    module.defs.iter().any(|def| {
        let Def::SelectRule(rule) = def else {
            return false;
        };
        let Some(pattern) = rule.patterns.first() else {
            return false;
        };
        let schema_source_def_field = pattern_args(pattern).and_then(|args| {
            pattern_schema_name(pattern).and_then(|_| infer_schema_source_def_field(args))
        });
        constructor_needs_source_defs(&rule.emit, final_inst_defs, schema_source_def_field, true)
    })
}

fn constructor_needs_source_defs(
    constructor: &Constructor,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    schema_source_def_field: Option<&str>,
    preserve_operands: bool,
) -> bool {
    let Constructor::Inst { opcode, args } = constructor else {
        return false;
    };
    if opcode == "seq" {
        return args.iter().any(|arg| {
            constructor_needs_source_defs(arg, final_inst_defs, schema_source_def_field, false)
        });
    }

    let Some(inst_def) = final_inst_defs.get(opcode) else {
        return false;
    };
    let use_direct_schema_def = preserve_operands
        && schema_source_def_field.is_some()
        && inst_def_def_operand_count(&inst_def.operands) == 1;

    let mut explicit_arg_cursor = 0usize;
    for (operand_index, operand) in inst_def.operands.iter().enumerate() {
        let use_source_def = preserve_operands
            && matches!(
                operand,
                OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
            )
            && args.len() - explicit_arg_cursor
                == min_explicit_args_from(&inst_def.operands, operand_index);
        if use_source_def {
            return !use_direct_schema_def;
        }
        explicit_arg_cursor += 1;
    }

    false
}

// =============================================================================
// 编译阶段函数
// =============================================================================

/// 解析输入
fn parse_input(input: &str) -> Result<crate::ast::Module, String> {
    match parser::parse(input) {
        Ok(m) => Ok(m),
        Err(e) => {
            use miette::Diagnostic;
            let report = if e.source_code().is_some() {
                miette::Report::new(e)
            } else {
                miette::Report::new(e).with_source_code(input.to_string())
            };
            Err(format!("{:?}", report))
        }
    }
}

#[derive(Debug, Clone)]
struct FinalInstDef {
    operands: Vec<OperandConstraint>,
    implicit_uses: Vec<String>,
    implicit_defs: Vec<String>,
    clobbers: Vec<String>,
    emit: Vec<EmitExpr>,
    is_pseudo: bool,
}

/// 收集定义
fn collect_definitions(
    module: &crate::ast::Module,
) -> (
    HashMap<String, ExtractorDef>,
    HashMap<String, crate::ast::TemplateDef>,
    HashMap<String, MacroDef>,
) {
    let mut extractors = HashMap::new();
    let mut templates = HashMap::new();
    let mut macros = HashMap::new();

    for def in &module.defs {
        match def {
            Def::Extractor(exc) => {
                extractors.insert(exc.name.clone(), exc.clone());
            }
            Def::Template(t) => {
                templates.insert(t.name.clone(), t.clone());
            }
            Def::Macro(m) => {
                macros.insert(m.name.clone(), m.clone());
            }
            _ => {}
        }
    }

    (extractors, templates, macros)
}

/// 替换表达式中的参数
fn subst_expr(expr: &Expr, args_map: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Variable(v) => args_map.get(v).cloned().unwrap_or_else(|| expr.clone()),
        Expr::HwEnc(_)
        | Expr::SlotBaseHwEnc(_)
        | Expr::SlotOffset(_)
        | Expr::SlotSize(_)
        | Expr::SlotAlign(_)
        | Expr::Int(_) => expr.clone(),
        Expr::BitOr(a, b) => Expr::BitOr(
            Box::new(subst_expr(a, args_map)),
            Box::new(subst_expr(b, args_map)),
        ),
        Expr::BitAnd(a, b) => Expr::BitAnd(
            Box::new(subst_expr(a, args_map)),
            Box::new(subst_expr(b, args_map)),
        ),
        Expr::Shl(a, b) => Expr::Shl(
            Box::new(subst_expr(a, args_map)),
            Box::new(subst_expr(b, args_map)),
        ),
        Expr::Shr(a, b) => Expr::Shr(
            Box::new(subst_expr(a, args_map)),
            Box::new(subst_expr(b, args_map)),
        ),
        Expr::Call(name, args) => {
            let new_args = args.iter().map(|a| subst_expr(a, args_map)).collect();
            Expr::Call(name.clone(), new_args)
        }
    }
}

/// 替换发射表达式中的参数
fn subst_emit(emit: &EmitExpr, args_map: &HashMap<String, Expr>) -> EmitExpr {
    match emit {
        EmitExpr::ByteExpr(e) => EmitExpr::ByteExpr(Box::new(subst_expr(e, args_map))),
        EmitExpr::Imm16(e) => EmitExpr::Imm16(Box::new(subst_expr(e, args_map))),
        EmitExpr::Imm32(e) => EmitExpr::Imm32(Box::new(subst_expr(e, args_map))),
        EmitExpr::Imm64(e) => EmitExpr::Imm64(Box::new(subst_expr(e, args_map))),
        EmitExpr::Rel32(name) => EmitExpr::Rel32(match args_map.get(name) {
            Some(Expr::Variable(nv)) => nv.clone(),
            _ => name.clone(),
        }),
        EmitExpr::If(cond, then_p, else_p) => EmitExpr::If(
            Box::new(subst_expr(cond, args_map)),
            then_p.iter().map(|e| subst_emit(e, args_map)).collect(),
            else_p.iter().map(|e| subst_emit(e, args_map)).collect(),
        ),
        _ => emit.clone(),
    }
}

/// 替换 operand constraint 中的变量名
fn subst_operand(op: &OperandConstraint, args_map: &HashMap<String, Expr>) -> OperandConstraint {
    let subst_name = |name: &str| -> String {
        match args_map.get(name) {
            Some(Expr::Variable(nv)) => nv.clone(),
            _ => name.to_string(),
        }
    };

    match op {
        OperandConstraint::Use(v) => OperandConstraint::Use(subst_name(v)),
        OperandConstraint::FixedUse { reg, src } => OperandConstraint::FixedUse {
            reg: reg.clone(),
            src: subst_name(src),
        },
        OperandConstraint::Def(v) => OperandConstraint::Def(subst_name(v)),
        OperandConstraint::Imm(v) => OperandConstraint::Imm(subst_name(v)),
        OperandConstraint::Block(v) => OperandConstraint::Block(subst_name(v)),
        OperandConstraint::Global(v) => OperandConstraint::Global(subst_name(v)),
        OperandConstraint::StackSlot(v) => OperandConstraint::StackSlot(subst_name(v)),
        OperandConstraint::TiedDef { dst, src } => OperandConstraint::TiedDef {
            dst: subst_name(dst),
            src: subst_name(src),
        },
    }
}

/// 实例化模板
fn instantiate_templates(
    module: &crate::ast::Module,
    templates: &HashMap<String, crate::ast::TemplateDef>,
) -> HashMap<String, FinalInstDef> {
    let mut final_inst_defs = HashMap::new();

    for def in &module.defs {
        match def {
            Def::Inst(inst) => {
                let mut inst = inst.clone();

                if let Some(t_inst) = &inst.template {
                    if let Some(t_def) = templates.get(&t_inst.name) {
                        let args_map: HashMap<_, _> = t_def
                            .args
                            .iter()
                            .enumerate()
                            .filter_map(|(i, name)| {
                                t_inst.args.get(i).map(|val| (name.clone(), val.clone()))
                            })
                            .collect();

                        if inst.operands.is_empty() {
                            inst.operands = t_def
                                .operands
                                .iter()
                                .map(|op| subst_operand(op, &args_map))
                                .collect();
                        }

                        if inst.emit.is_empty() {
                            inst.emit = t_def
                                .emit
                                .iter()
                                .map(|e| subst_emit(e, &args_map))
                                .collect();
                        }

                        if inst.clobbers.is_empty() {
                            inst.clobbers = t_def.clobbers.clone();
                        }
                        if inst.implicit_uses.is_empty() {
                            inst.implicit_uses = t_def.implicit_uses.clone();
                        }
                        if inst.implicit_defs.is_empty() {
                            inst.implicit_defs = t_def.implicit_defs.clone();
                        }
                    }
                }

                final_inst_defs.insert(
                    inst.name.clone(),
                    FinalInstDef {
                        operands: inst.operands,
                        implicit_uses: inst.implicit_uses,
                        implicit_defs: inst.implicit_defs,
                        clobbers: inst.clobbers,
                        emit: inst.emit,
                        is_pseudo: false,
                    },
                );
            }
            Def::PseudoInst(inst) => {
                final_inst_defs.insert(
                    inst.name.clone(),
                    FinalInstDef {
                        operands: inst.operands.clone(),
                        implicit_uses: inst.implicit_uses.clone(),
                        implicit_defs: inst.implicit_defs.clone(),
                        clobbers: inst.clobbers.clone(),
                        emit: Vec::new(),
                        is_pseudo: true,
                    },
                );
            }
            _ => {}
        }
    }

    final_inst_defs
}

/// 生成文件头
fn generate_header(
    output: &mut String,
    arch: &str,
    needs_positional_helpers: bool,
    needs_source_defs_helper: bool,
) {
    writeln!(
        output,
        "// Generated by veloc-isle (arch: {}). DO NOT EDIT.",
        arch
    )
    .unwrap();
    writeln!(output).unwrap();
    writeln!(
        output,
        r#"use crate::mir::{{MachineInst, MachineOperand, Reg}};
use smallvec::SmallVec;
use crate::target::arch::{{
    AbiDescriptor, AbiPreservedSet, AbiRegisterPool, AbiStackDescriptor, AbiValueClass,
    CpuDescription, FixedUseConstraint, GenericInstMetadata, LoweringContext,
    PreIselRewriteRuleData, RegInfo,
    SelectResult, TargetArch, TargetInstMetadata, TargetTiedOperandMetadata, TiedOperandConstraint,
}};
pub use veloc_ir::Type;

trait IntoOptReg {{
    fn into_opt_reg(self) -> Option<Reg>;
}}

impl IntoOptReg for Reg {{
    fn into_opt_reg(self) -> Option<Reg> {{
        Some(self)
    }}
}}

impl IntoOptReg for Option<Reg> {{
    fn into_opt_reg(self) -> Option<Reg> {{
        self
    }}
}}

fn reg_value<R: IntoOptReg>(value: R) -> Option<Reg> {{
    value.into_opt_reg()
}}

fn reg_value_to_vreg<R: IntoOptReg>(value: R) -> Option<crate::mir::VReg> {{
    value
        .into_opt_reg()
        .and_then(|reg| reg.is_vreg().then(|| crate::mir::VReg::from_u32(reg.index())))
}}

"#
    )
    .unwrap();
    if needs_source_defs_helper {
        writeln!(
            output,
            r#"
fn source_defs(inst: &MachineInst) -> SmallVec<[Reg; 2]> {{
    let mut defs = SmallVec::<[Reg; 2]>::new();
    for op in &inst.operands {{
        match op {{
            MachineOperand::Def(w) | MachineOperand::TiedDefUse(w) => defs.push(w.to_reg()),
            _ => {{}}
        }}
    }}
    defs
}}
"#
        )
        .unwrap();
    }
    if needs_positional_helpers {
        writeln!(
            output,
            r#"
fn operand_by_index(inst: &MachineInst, index: usize) -> Option<MachineOperand> {{
    inst.operands.get(index).cloned()
}}

fn vreg_by_index(inst: &MachineInst, index: usize) -> Option<crate::mir::VReg> {{
    let reg = inst.operands.get(index)?.as_reg()?;
    reg.is_vreg().then(|| crate::mir::VReg::from_u32(reg.index()))
}}
"#
        )
        .unwrap();
    }
    writeln!(output).unwrap();
}

/// 生成 TargetInst enum
fn generate_enum(output: &mut String, final_inst_defs: &HashMap<String, FinalInstDef>) {
    let mut sorted_insts: Vec<_> = final_inst_defs.keys().cloned().collect();
    sorted_insts.sort();

    writeln!(
        output,
        r#"
/// 目标架构指令特化
pub enum TargetInst {{"#
    )
    .unwrap();
    for name in &sorted_insts {
        writeln!(output, "    {},", name).unwrap();
    }
    writeln!(output, "}}").unwrap();
}

/// 生成 TargetInst 的 from_u32/as_u32 方法
fn generate_enum_conversions(output: &mut String, final_inst_defs: &HashMap<String, FinalInstDef>) {
    let mut sorted_insts: Vec<_> = final_inst_defs.keys().cloned().collect();
    sorted_insts.sort();

    writeln!(
        output,
        r#"
impl TargetInst {{
    pub fn from_u32(op: u32) -> Self {{
        match op {{"#
    )
    .unwrap();
    for (i, name) in sorted_insts.iter().enumerate() {
        writeln!(output, "            {} => Self::{},", i + 1, name).unwrap();
    }
    writeln!(
        output,
        r#"            _ => panic!("Unknown opcode: {{}}", op),
        }}
    }}

    pub fn as_u32(&self) -> u32 {{
        match self {{"#
    )
    .unwrap();
    for (i, name) in sorted_insts.iter().enumerate() {
        writeln!(output, "            Self::{} => {},", name, i + 1).unwrap();
    }
    writeln!(output, "        }}\n    }}\n}}").unwrap();
}

fn reg_const_name(name: &str) -> String {
    format!("REG_{}", sanitize_ident(name).to_ascii_uppercase())
}

fn format_slice(entries: Vec<String>) -> String {
    if entries.is_empty() {
        "&[]".to_string()
    } else {
        format!("&[{}]", entries.join(", "))
    }
}

fn format_tied_metadata_slice(operands: &[OperandConstraint]) -> String {
    let entries = operands
        .iter()
        .enumerate()
        .filter(|(_, operand)| matches!(operand, OperandConstraint::TiedDef { .. }))
        .map(|(index, _)| format!("TargetTiedOperandMetadata {{ operand: {} }}", index))
        .collect();
    format_slice(entries)
}

fn format_fixed_use_slice(
    operands: &[OperandConstraint],
    reg_names: &BTreeSet<String>,
    inst_name: &str,
) -> String {
    let entries = operands
        .iter()
        .enumerate()
        .filter_map(|(index, operand)| match operand {
            OperandConstraint::FixedUse { reg, .. } => {
                if !reg_names.contains(reg) {
                    panic!(
                        "instruction {} references unknown fixed register {}",
                        inst_name, reg
                    );
                }
                Some(format!(
                    "FixedUseConstraint {{ use_operand: {}, reg: {} }}",
                    index,
                    reg_const_name(reg)
                ))
            }
            _ => None,
        })
        .collect();
    format_slice(entries)
}

fn format_reg_metadata_slice(
    regs: &[String],
    reg_names: &BTreeSet<String>,
    inst_name: &str,
    kind: &str,
) -> String {
    let entries = regs
        .iter()
        .map(|reg| {
            if !reg_names.contains(reg) {
                panic!(
                    "instruction {} references unknown {} register {}",
                    inst_name, kind, reg
                );
            }
            reg_const_name(reg)
        })
        .collect();
    format_slice(entries)
}

fn format_clobber_slice(clobbers: &[String]) -> String {
    let entries = clobbers
        .iter()
        .map(|clobber| format!("\"{}\"", clobber))
        .collect();
    format_slice(entries)
}

fn generate_target_inst_metadata(
    output: &mut String,
    module: &crate::ast::Module,
    final_inst_defs: &HashMap<String, FinalInstDef>,
) {
    let reg_names: BTreeSet<String> = module
        .defs
        .iter()
        .filter_map(|def| match def {
            Def::Reg(reg) => Some(reg.name.clone()),
            _ => None,
        })
        .collect();

    let mut sorted_insts: Vec<_> = final_inst_defs.keys().cloned().collect();
    sorted_insts.sort();

    writeln!(
        output,
        "\n/// Target instruction metadata generated from `def-inst` / `def-pseudo-inst`."
    )
    .unwrap();
    for name in &sorted_insts {
        let inst_def = final_inst_defs
            .get(name)
            .unwrap_or_else(|| panic!("missing final inst def for {}", name));
        let const_name = format!(
            "TARGET_INST_{}_METADATA",
            sanitize_ident(name).to_ascii_uppercase()
        );
        writeln!(
            output,
            "pub const {const_name}: TargetInstMetadata = TargetInstMetadata {{"
        )
        .unwrap();
        writeln!(
            output,
            "    tied_operands: {},",
            format_tied_metadata_slice(&inst_def.operands)
        )
        .unwrap();
        writeln!(
            output,
            "    fixed_uses: {},",
            format_fixed_use_slice(&inst_def.operands, &reg_names, name)
        )
        .unwrap();
        writeln!(
            output,
            "    implicit_uses: {},",
            format_reg_metadata_slice(&inst_def.implicit_uses, &reg_names, name, "implicit-use")
        )
        .unwrap();
        writeln!(
            output,
            "    implicit_defs: {},",
            format_reg_metadata_slice(&inst_def.implicit_defs, &reg_names, name, "implicit-def")
        )
        .unwrap();
        writeln!(
            output,
            "    clobbers: {},",
            format_clobber_slice(&inst_def.clobbers)
        )
        .unwrap();
        writeln!(output, "}};").unwrap();
    }

    writeln!(
        output,
        "\npub fn target_inst_metadata(opcode: TargetInst) -> &'static TargetInstMetadata {{"
    )
    .unwrap();
    writeln!(output, "    match opcode {{").unwrap();
    for name in &sorted_insts {
        let const_name = format!(
            "TARGET_INST_{}_METADATA",
            sanitize_ident(name).to_ascii_uppercase()
        );
        writeln!(
            output,
            "        TargetInst::{name} => &{const_name},",
            name = name,
            const_name = const_name
        )
        .unwrap();
    }
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DerivedGenericInstMetadata {
    tied_operands: Vec<(usize, usize)>,
    commute_operand_pairs: Vec<(usize, usize)>,
    fixed_uses: Vec<(usize, String)>,
}

fn is_commutative_generic_opcode(opcode: &str) -> bool {
    matches!(opcode, "G_ADD" | "G_MUL" | "G_AND" | "G_OR" | "G_XOR")
}

fn schema_field_operand_index(schema: &str, field: &str) -> Option<usize> {
    match schema {
        "UnaryReg" => match field {
            "dst" => Some(0),
            "src" => Some(1),
            _ => None,
        },
        "BinaryReg" => match field {
            "dst" => Some(0),
            "lhs" => Some(1),
            "rhs" => Some(2),
            _ => None,
        },
        "Load" => match field {
            "dst" => Some(0),
            "base" => Some(1),
            _ => None,
        },
        "LoadOffset" => match field {
            "dst" => Some(0),
            "base" => Some(1),
            "offset" => Some(2),
            _ => None,
        },
        "Store" => match field {
            "src" => Some(0),
            "base" => Some(1),
            _ => None,
        },
        "StoreOffset" => match field {
            "src" => Some(0),
            "base" => Some(1),
            "offset" => Some(2),
            _ => None,
        },
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
        let use_source_def = matches!(
            operand,
            OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
        ) && schema_source_def_field.is_some()
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
    module: &crate::ast::Module,
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

        if target_inst_def
            .operands
            .iter()
            .any(|operand| matches!(operand, OperandConstraint::TiedDef { .. }))
        {
            if let (Some(def_operand), Some(use_operand)) = (
                schema_field_operand_index(schema_name, "dst"),
                schema_field_operand_index(schema_name, "lhs"),
            ) {
                let pair = (def_operand, use_operand);
                if !metadata.tied_operands.contains(&pair) {
                    metadata.tied_operands.push(pair);
                }
            }
        }

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

        if is_commutative_generic_opcode(opcode) {
            let pair = (1usize, 2usize);
            if !metadata.commute_operand_pairs.contains(&pair) {
                metadata.commute_operand_pairs.push(pair);
            }
        }
    }

    for metadata in result.values_mut() {
        metadata.tied_operands.sort_unstable();
        metadata.tied_operands.dedup();
        metadata.commute_operand_pairs.sort_unstable();
        metadata.commute_operand_pairs.dedup();
        metadata.fixed_uses.sort_unstable();
        metadata.fixed_uses.dedup();
    }

    result.retain(|_, metadata| {
        !(metadata.tied_operands.is_empty()
            && metadata.commute_operand_pairs.is_empty()
            && metadata.fixed_uses.is_empty())
    });

    result
}

fn generate_generic_inst_metadata(
    output: &mut String,
    module: &crate::ast::Module,
    final_inst_defs: &HashMap<String, FinalInstDef>,
) {
    let metadata_map = derive_generic_inst_metadata(module, final_inst_defs);

    if metadata_map.is_empty() {
        writeln!(
            output,
            "\npub fn generic_inst_metadata(_opcode: crate::mir::GenericOpcode) -> &'static GenericInstMetadata {{\n    &GenericInstMetadata::EMPTY\n}}"
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
        let tied_entries = metadata
            .tied_operands
            .iter()
            .map(|(def_operand, use_operand)| {
                format!(
                    "TiedOperandConstraint {{ def_operand: {}, use_operand: {} }}",
                    def_operand, use_operand
                )
            })
            .collect();
        let commute_entries = metadata
            .commute_operand_pairs
            .iter()
            .map(|(lhs, rhs)| format!("({}, {})", lhs, rhs))
            .collect();
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
        writeln!(output, "    tied_operands: {},", format_slice(tied_entries)).unwrap();
        writeln!(
            output,
            "    commute_operand_pairs: {},",
            format_slice(commute_entries)
        )
        .unwrap();
        writeln!(output, "    fixed_uses: {},", format_slice(fixed_entries)).unwrap();
        writeln!(output, "}};").unwrap();
    }

    writeln!(
        output,
        "\npub fn generic_inst_metadata(opcode: crate::mir::GenericOpcode) -> &'static GenericInstMetadata {{"
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
            "        crate::mir::GenericOpcode::{opcode} => &{const_name},",
            opcode = opcode,
            const_name = const_name
        )
        .unwrap();
    }
    writeln!(output, "        _ => &GenericInstMetadata::EMPTY,").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
}

/// 生成 emit 方法
fn generate_emit_method(
    output: &mut String,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    macros: &HashMap<String, MacroDef>,
) {
    let mut sorted_insts: Vec<_> = final_inst_defs.keys().cloned().collect();
    sorted_insts.sort();

    writeln!(
        output,
        r#"
impl TargetInst {{
    pub fn emit<E: crate::target::arch::TargetEmitter>(
        &self,
        emitter: &mut crate::Emitter,
        inst: &MachineInst,
        mfunc: &crate::mir::MachineFunction,
    ) -> Result<(), crate::error::Error> {{
        match self {{"#
    )
    .unwrap();

    for name in &sorted_insts {
        writeln!(output, "            TargetInst::{} => {{", name).unwrap();
        if let Some(inst_def) = final_inst_defs.get(name) {
            if inst_def.is_pseudo {
                writeln!(
                    output,
                    "                return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!(\"Pseudo instruction {} must be lowered before emission\")));",
                    name
                )
                .unwrap();
            } else {
                for emit_expr in &inst_def.emit {
                    let code = generate_emit_expr(emit_expr, &inst_def.operands, macros);
                    writeln!(output, "                {}", code).unwrap();
                }
            }
        }
        writeln!(output, "                Ok(())\n            }}").unwrap();
    }

    writeln!(output, "        }}\n    }}\n}}").unwrap();
}

fn generate_cpu_info(output: &mut String, module: &crate::ast::Module) {
    let mut features: Vec<&crate::ast::FeatureDef> = Vec::new();
    let mut cpus: Vec<&crate::ast::CpuDef> = Vec::new();

    for def in &module.defs {
        if let Def::Feature(f) = def {
            features.push(f);
        } else if let Def::Cpu(c) = def {
            cpus.push(c);
        }
    }

    if features.is_empty() && cpus.is_empty() {
        return;
    }

    writeln!(output, "/// CPU descriptions generated from `def-cpu`.").unwrap();
    for cpu in &cpus {
        let prefix = sanitize_ident(&format!("CPU_{}", cpu.name)).to_ascii_uppercase();
        let feats = cpu
            .features
            .iter()
            .map(|f| format!("\"{}\"", f))
            .collect::<Vec<_>>()
            .join(", ");
        let limits = cpu
            .limitations
            .iter()
            .map(|f| format!("\"{}\"", f))
            .collect::<Vec<_>>()
            .join(", ");

        writeln!(
            output,
            "pub const {prefix}_FEATURES: &[&str] = &[{feats}];",
            prefix = prefix,
            feats = feats
        )
        .unwrap();
        writeln!(
            output,
            "pub const {prefix}_LIMITATIONS: &[&str] = &[{limits}];",
            prefix = prefix,
            limits = limits
        )
        .unwrap();
        writeln!(
            output,
            "pub const {prefix}_DESC: CpuDescription = CpuDescription {{ name: \"{name}\", features: {prefix}_FEATURES, limitations: {prefix}_LIMITATIONS }};",
            prefix = prefix,
            name = cpu.name
        )
        .unwrap();
    }
    writeln!(output, "pub const SUPPORTED_CPUS: &[CpuDescription] = &[").unwrap();
    for cpu in &cpus {
        let prefix = sanitize_ident(&format!("CPU_{}", cpu.name)).to_ascii_uppercase();
        writeln!(output, "    {prefix}_DESC,", prefix = prefix).unwrap();
    }
    writeln!(output, "];").unwrap();
    writeln!(output).unwrap();
}

/// 生成指令选择函数
fn generate_select_instruction(
    output: &mut String,
    module: &crate::ast::Module,
    extractors: &HashMap<String, ExtractorDef>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    arch: &str,
) {
    let reg_map = collect_reg_encs(module);
    let needs_positional_helpers = module_has_positional_rules(module);

    let mut decls = HashMap::new();
    for def in &module.defs {
        if let Def::Decl(decl) = def {
            decls.insert(decl.name.clone(), decl.clone());
        }
    }

    let extra_bound = if arch == "x86_64" {
        " + crate::target::x86_64::lowering::X86LoweringContext"
    } else {
        ""
    };

    writeln!(
        output,
        r#"
pub fn select_instructions<C: LoweringContext{extra_bound}>(
    ctx: &C,
    inst: &MachineInst,
    out: &mut alloc::vec::Vec<MachineInst>,
) -> Result<SelectResult, crate::error::Error> {{
    use crate::mir::{{GenericOpcode, MachineOpcode, VReg}};
    use crate::target::arch::SelectResult;
    use crate::target::{arch}::isle::TargetInst;

    let decoded = inst.decode_generic().ok();
"#,
        extra_bound = extra_bound,
    )
    .unwrap();
    if needs_positional_helpers {
        writeln!(output, "    let v0 = vreg_by_index(inst, 0);").unwrap();
        writeln!(output, "    let v1 = vreg_by_index(inst, 1);").unwrap();
        writeln!(output, "    let v2 = vreg_by_index(inst, 2);").unwrap();
    }
    writeln!(output).unwrap();

    // 为每个 extractor 生成闭包以保持代码简洁
    for (name, exc) in extractors {
        let cond = generate_pattern_condition(&exc.body, "v", &decls);
        writeln!(
            output,
            "    let is_{} = |v_opt: Option<VReg>| v_opt.map_or(false, |v| {});",
            name.to_lowercase(),
            cond
        )
        .unwrap();
    }
    writeln!(output).unwrap();

    writeln!(
        output,
        "    let MachineOpcode::Generic(opcode) = &inst.opcode else {{"
    )
    .unwrap();
    writeln!(output, "        return Ok(SelectResult::Keep);").unwrap();
    writeln!(output, "    }};").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "    // 尝试按规则选择指令序列").unwrap();
    writeln!(output, "    match opcode {{").unwrap();

    // 预处理规则，按操作码分组
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
                    "            if let Some(crate::mir::DecodedGenericInst::{}({})) = decoded.as_ref() {{",
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
                    let var_map = collect_var_bindings(grouped_pattern);
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
                    emit_constructor_sequence(
                        output,
                        &grouped_rule.emit,
                        &var_map,
                        final_inst_defs,
                        &reg_map,
                        Some(schema_var),
                        infer_schema_source_def_field(grouped_args),
                    );
                    writeln!(output, "                }}").unwrap();
                    rule_index += 1;
                }

                writeln!(output, "            }}").unwrap();
            } else {
                let var_map = collect_var_bindings(pattern);
                let conditions = collect_positional_rule_conditions(args, extractors);

                if conditions.is_empty() {
                    writeln!(output, "            {{").unwrap();
                } else {
                    writeln!(output, "            if {} {{", conditions.join(" && ")).unwrap();
                }

                emit_constructor_sequence(
                    output,
                    &rule.emit,
                    &var_map,
                    final_inst_defs,
                    &reg_map,
                    None,
                    None,
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
        "    Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"No matching ISLE rule found for instruction\")))"
    )
    .unwrap();
    writeln!(output, "}}").unwrap();
}

fn rewrite_rule_view<'a>(def: &'a Def) -> Option<(&'a RuleAttrs, &'a [Pattern], &'a Constructor)> {
    match def {
        Def::RewriteRule(rule) => Some((&rule.attrs, &rule.patterns, &rule.replace)),
        _ => None,
    }
}

fn collect_rewrite_vars(pattern: &Pattern, slots: &mut HashMap<String, u32>, next_slot: &mut u32) {
    match pattern.strip_node_binds() {
        Pattern::Variable(name) => {
            slots.entry(name.clone()).or_insert_with(|| {
                let slot = *next_slot;
                *next_slot += 1;
                slot
            });
        }
        Pattern::Opcode { args, .. } | Pattern::Schema { args, .. } => {
            for arg in args {
                match arg {
                    PatternArg::Positional(p) => collect_rewrite_vars(p, slots, next_slot),
                    PatternArg::Named { pattern, .. } => {
                        collect_rewrite_vars(pattern, slots, next_slot)
                    }
                }
            }
        }
        Pattern::And(parts) => {
            for part in parts {
                collect_rewrite_vars(part, slots, next_slot);
            }
        }
        Pattern::StackSlot(inner) | Pattern::NodeBind { inner, .. } => {
            collect_rewrite_vars(inner, slots, next_slot)
        }
        Pattern::IntConst(_) | Pattern::Block(_) | Pattern::CondCode(_) => {}
    }
}

fn render_pre_isel_expr(pattern: &Pattern, slots: &HashMap<String, u32>) -> Option<String> {
    match pattern.strip_node_binds() {
        Pattern::Variable(name) => Some(format!(
            "PreIselRewriteExpr::Var({})",
            slots.get(name).copied().unwrap_or(0)
        )),
        Pattern::IntConst(i) => Some(format!("PreIselRewriteExpr::Imm({})", i)),
        Pattern::Opcode { opcode, args, .. } | Pattern::Schema { opcode, args, .. } => {
            let mut rendered = Vec::new();
            for arg in args {
                let arg = match arg {
                    PatternArg::Positional(p) => p,
                    PatternArg::Named { pattern, .. } => pattern.as_ref(),
                };
                rendered.push(render_pre_isel_expr(arg, slots)?);
            }
            if rendered.is_empty() {
                Some(format!(
                    "PreIselRewriteExpr::Op {{ opcode: crate::mir::GenericOpcode::{}, args: &[] }}",
                    opcode
                ))
            } else {
                Some(format!(
                    "PreIselRewriteExpr::Op {{ opcode: crate::mir::GenericOpcode::{}, args: &[{}] }}",
                    opcode,
                    rendered.join(", ")
                ))
            }
        }
        Pattern::StackSlot(inner) | Pattern::NodeBind { inner, .. } => {
            render_pre_isel_expr(inner, slots)
        }
        Pattern::And(_) | Pattern::Block(_) | Pattern::CondCode(_) => None,
    }
}

fn render_pre_isel_constructor(
    constructor: &Constructor,
    slots: &HashMap<String, u32>,
) -> Option<String> {
    match constructor {
        Constructor::Variable(name) => Some(format!(
            "PreIselRewriteExpr::Var({})",
            slots.get(name).copied().unwrap_or(0)
        )),
        Constructor::Imm(i) => Some(format!("PreIselRewriteExpr::Imm({})", i)),
        Constructor::Reg(_) => None,
        Constructor::Inst { opcode, args } => {
            if opcode == "seq" {
                return None;
            }
            let mut rendered = Vec::new();
            for arg in args {
                rendered.push(render_pre_isel_constructor(arg, slots)?);
            }
            if rendered.is_empty() {
                Some(format!(
                    "PreIselRewriteExpr::Op {{ opcode: crate::mir::GenericOpcode::{}, args: &[] }}",
                    opcode
                ))
            } else {
                Some(format!(
                    "PreIselRewriteExpr::Op {{ opcode: crate::mir::GenericOpcode::{}, args: &[{}] }}",
                    opcode,
                    rendered.join(", ")
                ))
            }
        }
    }
}

fn generate_pre_isel_rewrite_tables(output: &mut String, module: &crate::ast::Module) {
    let mut rules = Vec::new();
    for def in &module.defs {
        let Def::RewriteRule(rule) = def else {
            continue;
        };
        let Some(pattern) = rule.patterns.first() else {
            continue;
        };
        let mut slots = HashMap::new();
        let mut next_slot = 0u32;
        collect_rewrite_vars(pattern, &mut slots, &mut next_slot);
        let Some(match_expr) = render_pre_isel_expr(pattern, &slots) else {
            continue;
        };
        let Some(replace_expr) = render_pre_isel_constructor(&rule.replace, &slots) else {
            continue;
        };
        let name = format!("rewrite_{}", rules.len());
        rules.push((
            name,
            match_expr,
            replace_expr,
            rule.attrs.cost.unwrap_or(0),
            rule.attrs.priority.unwrap_or(0),
        ));
    }

    writeln!(
        output,
        "\npub static PRE_ISEL_REWRITE_RULES: &[PreIselRewriteRuleData] = &["
    )
    .unwrap();
    for (name, match_expr, replace_expr, cost, priority) in rules {
        writeln!(
            output,
            "    PreIselRewriteRuleData {{ name: \"{}\", match_expr: {}, replace_expr: {}, cost: {}, priority: {} }},",
            name,
            match_expr,
            replace_expr,
            cost,
            priority
        )
        .unwrap();
    }
    writeln!(output, "];").unwrap();
}

fn generate_rewrite_instruction(
    output: &mut String,
    module: &crate::ast::Module,
    extractors: &HashMap<String, ExtractorDef>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    arch: &str,
) {
    let reg_map = collect_reg_encs(module);
    let needs_positional_helpers = module_has_positional_rules(module);

    let mut decls = HashMap::new();
    for def in &module.defs {
        if let Def::Decl(decl) = def {
            decls.insert(decl.name.clone(), decl.clone());
        }
    }

    let extra_bound = if arch == "x86_64" {
        " + crate::target::x86_64::lowering::X86LoweringContext"
    } else {
        ""
    };

    writeln!(
        output,
        r#"
pub fn rewrite_instructions<C: LoweringContext{extra_bound}>(
    ctx: &C,
    inst: &MachineInst,
    out: &mut alloc::vec::Vec<MachineInst>,
) -> Result<RewriteResult, crate::error::Error> {{
    use crate::mir::{{GenericOpcode, MachineOpcode, VReg}};
    use crate::target::arch::RewriteResult;
    use crate::target::{arch}::isle::TargetInst;

    let decoded = inst.decode_generic().ok();
"#,
        extra_bound = extra_bound,
    )
    .unwrap();
    if needs_positional_helpers {
        writeln!(output, "    let v0 = vreg_by_index(inst, 0);").unwrap();
        writeln!(output, "    let v1 = vreg_by_index(inst, 1);").unwrap();
        writeln!(output, "    let v2 = vreg_by_index(inst, 2);").unwrap();
    }
    writeln!(output).unwrap();

    for (name, exc) in extractors {
        let cond = generate_pattern_condition(&exc.body, "v", &decls);
        writeln!(
            output,
            "    let is_{} = |v_opt: Option<VReg>| v_opt.map_or(false, |v| {});",
            name.to_lowercase(),
            cond
        )
        .unwrap();
    }
    writeln!(output).unwrap();

    writeln!(
        output,
        "    let MachineOpcode::Generic(opcode) = &inst.opcode else {{"
    )
    .unwrap();
    writeln!(output, "        return Ok(RewriteResult::Keep);").unwrap();
    writeln!(output, "    }};").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "    match opcode {{").unwrap();

    let mut opcode_rules: HashMap<String, Vec<&Def>> = HashMap::new();
    for def in &module.defs {
        if let Def::RewriteRule(rule) = def {
            if let Some(pattern) = rule.patterns.first() {
                if let Some(opcode) = pattern_opcode(pattern) {
                    opcode_rules
                        .entry(opcode.to_string())
                        .or_default()
                        .push(def);
                }
            }
        }
    }

    let mut opcodes: Vec<_> = opcode_rules.keys().cloned().collect();
    opcodes.sort();

    for op in opcodes {
        writeln!(output, "        GenericOpcode::{} => {{", op).unwrap();
        let rules = &opcode_rules[&op];
        for rule_def in rules {
            let Some((attrs, patterns, replace)) = rewrite_rule_view(rule_def) else {
                continue;
            };
            let Some(pattern) = patterns.first() else {
                continue;
            };
            let Some(args) = pattern_args(pattern) else {
                continue;
            };
            let _ = attrs;
            let var_map = collect_var_bindings(pattern);
            let schema_name = pattern_schema_name(pattern);
            let conditions = if let Some(schema_name) = schema_name {
                collect_schema_rule_conditions(args, extractors, "schema_inst", schema_name)
            } else {
                collect_positional_rule_conditions(args, extractors)
            };

            if let Some(schema_name) = schema_name {
                writeln!(
                    output,
                    "            if let Some(crate::mir::DecodedGenericInst::{}(schema_inst)) = decoded.as_ref() {{",
                    schema_name
                )
                .unwrap();
                if !conditions.is_empty() {
                    writeln!(output, "                if {} {{", conditions.join(" && ")).unwrap();
                } else {
                    writeln!(output, "                {{").unwrap();
                }
                emit_schema_value_bindings_for_constructor(output, replace, args, "schema_inst");
                emit_rewrite_constructor_sequence(
                    output,
                    replace,
                    &var_map,
                    final_inst_defs,
                    &reg_map,
                    Some("schema_inst"),
                    infer_schema_source_def_field(args),
                );
                writeln!(output, "                }}").unwrap();
                writeln!(output, "            }}").unwrap();
            } else {
                if !conditions.is_empty() {
                    writeln!(output, "            if {} {{", conditions.join(" && ")).unwrap();
                } else {
                    writeln!(output, "            {{").unwrap();
                }
                emit_rewrite_constructor_sequence(
                    output,
                    replace,
                    &var_map,
                    final_inst_defs,
                    &reg_map,
                    None,
                    None,
                );
                writeln!(output, "            }}").unwrap();
            }
        }
        writeln!(output, "        }}").unwrap();
    }

    writeln!(output, "        _ => {{}}").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "    Ok(RewriteResult::Keep)").unwrap();
    writeln!(output, "}}").unwrap();
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BindingSource {
    OperandIndex(usize),
    SchemaValue,
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
            if conds.is_empty() {
                None
            } else {
                Some(conds)
            }
        }
        Pattern::Opcode { opcode, .. } if extractors.contains_key(opcode) => Some(vec![format!(
            "is_{}(reg_value_to_vreg({}.{}))",
            opcode.to_lowercase(),
            schema_var,
            field
        )]),
        Pattern::IntConst(value) => Some(vec![format!("{}.{} == {}", schema_var, field, value)]),
        Pattern::CondCode(cc) => render_cond_code_match(schema_name, *cc).map(|expr| {
            if schema_name == "ICmp" {
                vec![format!("{}.{} == Some({})", schema_var, field, expr)]
            } else {
                vec![format!("{}.{} == {}", schema_var, field, expr)]
            }
        }),
        Pattern::Block(_) => None,
        Pattern::StackSlot(_) | Pattern::Opcode { .. } => None,
        Pattern::Schema { .. } => None,
        Pattern::NodeBind { .. } => None,
    }
}

fn render_cond_code_match(schema_name: &str, cc: CondCode) -> Option<&'static str> {
    match schema_name {
        "ICmp" => Some(match cc {
            CondCode::E => "veloc_ir::IntCC::Eq",
            CondCode::NE => "veloc_ir::IntCC::Ne",
            CondCode::L => "veloc_ir::IntCC::LtS",
            CondCode::LE => "veloc_ir::IntCC::LeS",
            CondCode::G => "veloc_ir::IntCC::GtS",
            CondCode::GE => "veloc_ir::IntCC::GeS",
            CondCode::B => "veloc_ir::IntCC::LtU",
            CondCode::BE => "veloc_ir::IntCC::LeU",
            CondCode::A => "veloc_ir::IntCC::GtU",
            CondCode::AE => "veloc_ir::IntCC::GeU",
        }),
        "FCmp" => Some(match cc {
            CondCode::E => "veloc_ir::FloatCC::Eq",
            CondCode::NE => "veloc_ir::FloatCC::Ne",
            CondCode::L | CondCode::B => "veloc_ir::FloatCC::Lt",
            CondCode::LE | CondCode::BE => "veloc_ir::FloatCC::Le",
            CondCode::G | CondCode::A => "veloc_ir::FloatCC::Gt",
            CondCode::GE | CondCode::AE => "veloc_ir::FloatCC::Ge",
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

fn emit_schema_value_bindings_for_constructor(
    output: &mut String,
    constructor: &Constructor,
    args: &[PatternArg],
    schema_var: &str,
) {
    let mut needed_vars = BTreeSet::new();
    collect_constructor_variables(constructor, &mut needed_vars);

    let mut bindings = BTreeMap::new();
    for (field, pattern) in named_args(args) {
        let mut vars = Vec::new();
        collect_pattern_variables(pattern, &mut vars);
        for var in vars {
            if needed_vars.contains(&var) {
                bindings.entry(var).or_insert_with(|| field.to_string());
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
            if !collect_schema_rule_conditions(args, extractors, "schema_inst", schema_name).is_empty()
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
    var_map: &HashMap<String, BindingSource>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    reg_map: &HashMap<String, u32>,
    schema_var: Option<&str>,
    schema_source_def_field: Option<&str>,
) {
    match constructor {
        Constructor::Inst { opcode, args } if opcode == "seq" => {
            for (i, c) in args.iter().enumerate() {
                emit_single_inst(
                    output,
                    c,
                    var_map,
                    final_inst_defs,
                    reg_map,
                    schema_var,
                    schema_source_def_field,
                    i,
                    false,
                    true,
                );
            }
            writeln!(output, "                return Ok(SelectResult::Replace);").unwrap();
        }
        _ => {
            emit_single_inst(
                output,
                constructor,
                var_map,
                final_inst_defs,
                reg_map,
                schema_var,
                schema_source_def_field,
                0,
                true,
                true,
            );
            writeln!(output, "                return Ok(SelectResult::InPlace);").unwrap();
        }
    }
}

fn emit_single_inst(
    output: &mut String,
    constructor: &Constructor,
    var_map: &HashMap<String, BindingSource>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    reg_map: &HashMap<String, u32>,
    schema_var: Option<&str>,
    schema_source_def_field: Option<&str>,
    index: usize,
    preserve_operands: bool,
    emit_to_out: bool,
) {
    let Constructor::Inst { opcode, args } = constructor else {
        writeln!(
            output,
            "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Invalid constructor\")));"
        )
        .unwrap();
        return;
    };

    if let Some(inst_def) = final_inst_defs.get(opcode) {
        let implicit_uses: Vec<u32> = inst_def
            .implicit_uses
            .iter()
            .filter_map(|r| reg_map.get(r).copied())
            .collect();
        let implicit_defs: Vec<u32> = inst_def
            .implicit_defs
            .iter()
            .filter_map(|r| reg_map.get(r).copied())
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
            "                let {}ops_{} = SmallVec::<[MachineOperand; 4]>::new();",
            ops_binding, index
        )
        .unwrap();
        let use_direct_schema_def = preserve_operands
            && schema_source_def_field.is_some()
            && inst_def_def_operand_count(&inst_def.operands) == 1;
        if preserve_operands
            && inst_def_has_def_operands(&inst_def.operands)
            && !use_direct_schema_def
        {
            writeln!(
                output,
                "                let source_defs_{index} = source_defs(inst);"
            )
            .unwrap();
            writeln!(
                output,
                "                let mut source_def_cursor_{index} = 0usize;"
            )
            .unwrap();
        }

        let mut explicit_arg_cursor = 0usize;
        for (operand_index, operand) in inst_def.operands.iter().enumerate() {
            let use_source_def = preserve_operands
                && matches!(
                    operand,
                    OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
                )
                && args.len() - explicit_arg_cursor
                    == min_explicit_args_from(&inst_def.operands, operand_index);

            if use_source_def {
                if let Some(field) = schema_source_def_field.filter(|_| use_direct_schema_def) {
                    emit_schema_source_def_operand(
                        output,
                        index,
                        operand,
                        schema_var.expect("schema source defs require a schema binding"),
                        field,
                    );
                } else {
                    emit_source_def_operand(output, index, operand);
                }
                continue;
            }

            let Some(arg) = args.get(explicit_arg_cursor) else {
                panic!(
                    "constructor {} is missing argument {} for target operand {}",
                    opcode, explicit_arg_cursor, operand_index
                );
            };
            emit_constructor_operand(output, index, arg, operand, var_map, reg_map);
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

        if !implicit_uses.is_empty() || !implicit_defs.is_empty() {
            writeln!(output, "                // Implicit uses/defs").unwrap();
            for r in &implicit_uses {
                let r_val = *r;
                writeln!(
                    output,
                    r#"                {{
                    let reg = Reg::new_preg({r_val});
                    if !ops_{index}.iter().any(|op| match op {{
                        MachineOperand::Use(r) => *r == reg,
                        MachineOperand::TiedDefUse(w) => w.to_reg() == reg,
                        _ => false,
                    }}) {{
                        ops_{index}.push(MachineOperand::Use(reg));
                    }}
                }}"#
                )
                .unwrap();
            }
            for r in &implicit_defs {
                let r_val = *r;
                writeln!(
                    output,
                    r#"                {{
                    let reg = Reg::new_preg({r_val});
                    if !ops_{index}.iter().any(|op| match op {{
                        MachineOperand::Def(w) => w.to_reg() == reg,
                        MachineOperand::TiedDefUse(w) => w.to_reg() == reg,
                        _ => false,
                    }}) {{
                        ops_{index}.push(MachineOperand::Def(crate::mir::Writable(reg)));
                    }}
                }}"#
                )
                .unwrap();
            }
        }

        writeln!(
            output,
            "                let inst_{} = MachineInst::build_generic(",
            index
        )
        .unwrap();
        if final_inst_defs.contains_key(opcode) {
            writeln!(
                output,
                "                    MachineOpcode::Target(TargetInst::{}.as_u32()),",
                opcode
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "                    MachineOpcode::Generic(GenericOpcode::{}),",
                opcode
            )
            .unwrap();
        }
        writeln!(output, "                    ops_{},", index).unwrap();
        writeln!(output, "                );").unwrap();
        if emit_to_out {
            writeln!(output, "                out.push(inst_{});", index).unwrap();
        }
    } else {
        writeln!(
            output,
            "                let mut ops_{} = SmallVec::<[MachineOperand; 4]>::new();",
            index
        )
        .unwrap();
        for arg in args {
            match arg {
                Constructor::Variable(name) => match var_map.get(name) {
                    Some(BindingSource::OperandIndex(idx)) => {
                        writeln!(
                            output,
                            "                ops_{}.push(operand_by_index(inst, {}).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand mapping failed\")))?);",
                            index, idx
                        )
                        .unwrap();
                    }
                    Some(BindingSource::SchemaValue) => {
                        writeln!(
                            output,
                            "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Schema value requires a target instruction schema\")));"
                        )
                        .unwrap();
                        return;
                    }
                    None => {
                        writeln!(
                            output,
                            "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Unknown constructor variable\")));"
                        )
                        .unwrap();
                        return;
                    }
                },
                Constructor::Imm(i) => {
                    writeln!(
                        output,
                        "                ops_{}.push(MachineOperand::Imm({}));",
                        index, i
                    )
                    .unwrap();
                }
                Constructor::Reg(name) => {
                    let enc = reg_map.get(name).copied().unwrap_or(0);
                    writeln!(
                        output,
                        "                ops_{}.push(MachineOperand::Use(Reg::new_preg({})));",
                        index, enc
                    )
                    .unwrap();
                }
                _ => {
                    writeln!(
                        output,
                        "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Unsupported constructor arg\")));"
                    )
                    .unwrap();
                    return;
                }
            }
        }

        writeln!(
            output,
            "                let inst_{} = MachineInst::build_generic(",
            index
        )
        .unwrap();
        writeln!(
            output,
            "                    MachineOpcode::Target(TargetInst::{}.as_u32()),",
            opcode
        )
        .unwrap();
        writeln!(output, "                    ops_{},", index).unwrap();
        writeln!(output, "                );").unwrap();
        if emit_to_out {
            writeln!(output, "                out.push(inst_{});", index).unwrap();
        }
    }
}

fn emit_rewrite_constructor_sequence(
    output: &mut String,
    constructor: &Constructor,
    var_map: &HashMap<String, BindingSource>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    reg_map: &HashMap<String, u32>,
    schema_var: Option<&str>,
    schema_source_def_field: Option<&str>,
) {
    match constructor {
        Constructor::Inst { opcode, args } if opcode == "seq" => {
            for (i, c) in args.iter().enumerate() {
                emit_rewrite_single_inst(
                    output,
                    c,
                    var_map,
                    final_inst_defs,
                    reg_map,
                    schema_var,
                    schema_source_def_field,
                    i,
                    false,
                    true,
                );
            }
            writeln!(output, "                return Ok(RewriteResult::Replace);").unwrap();
        }
        _ => {
            emit_rewrite_single_inst(
                output,
                constructor,
                var_map,
                final_inst_defs,
                reg_map,
                schema_var,
                schema_source_def_field,
                0,
                true,
                true,
            );
            writeln!(output, "                return Ok(RewriteResult::InPlace);").unwrap();
        }
    }
}

fn emit_rewrite_single_inst(
    output: &mut String,
    constructor: &Constructor,
    var_map: &HashMap<String, BindingSource>,
    final_inst_defs: &HashMap<String, FinalInstDef>,
    reg_map: &HashMap<String, u32>,
    schema_var: Option<&str>,
    schema_source_def_field: Option<&str>,
    index: usize,
    preserve_operands: bool,
    emit_to_out: bool,
) {
    let Constructor::Inst { opcode, args } = constructor else {
        writeln!(
            output,
            "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Invalid constructor\")));"
        )
        .unwrap();
        return;
    };

    if let Some(inst_def) = final_inst_defs.get(opcode) {
        let implicit_uses: Vec<u32> = inst_def
            .implicit_uses
            .iter()
            .filter_map(|r| reg_map.get(r).copied())
            .collect();
        let implicit_defs: Vec<u32> = inst_def
            .implicit_defs
            .iter()
            .filter_map(|r| reg_map.get(r).copied())
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
            "                let {}ops_{} = SmallVec::<[MachineOperand; 4]>::new();",
            ops_binding, index
        )
        .unwrap();
        let use_direct_schema_def = preserve_operands
            && schema_source_def_field.is_some()
            && inst_def_def_operand_count(&inst_def.operands) == 1;
        if preserve_operands
            && inst_def_has_def_operands(&inst_def.operands)
            && !use_direct_schema_def
        {
            writeln!(
                output,
                "                let source_defs_{index} = source_defs(inst);"
            )
            .unwrap();
            writeln!(
                output,
                "                let mut source_def_cursor_{index} = 0usize;"
            )
            .unwrap();
        }

        let mut explicit_arg_cursor = 0usize;
        for (operand_index, operand) in inst_def.operands.iter().enumerate() {
            let use_source_def = preserve_operands
                && matches!(
                    operand,
                    OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
                )
                && args.len() - explicit_arg_cursor
                    == min_explicit_args_from(&inst_def.operands, operand_index);

            if use_source_def {
                if let Some(field) = schema_source_def_field.filter(|_| use_direct_schema_def) {
                    emit_schema_source_def_operand(
                        output,
                        index,
                        operand,
                        schema_var.expect("schema source defs require a schema binding"),
                        field,
                    );
                } else {
                    emit_source_def_operand(output, index, operand);
                }
                continue;
            }

            let Some(arg) = args.get(explicit_arg_cursor) else {
                panic!(
                    "constructor {} is missing argument {} for target operand {}",
                    opcode, explicit_arg_cursor, operand_index
                );
            };
            emit_constructor_operand(output, index, arg, operand, var_map, reg_map);
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

        if !implicit_uses.is_empty() || !implicit_defs.is_empty() {
            writeln!(output, "                // Implicit uses/defs").unwrap();
            for r in &implicit_uses {
                let r_val = *r;
                writeln!(
                    output,
                    r#"                {{
                    let reg = Reg::new_preg({r_val});
                    if !ops_{index}.iter().any(|op| match op {{
                        MachineOperand::Use(r) => *r == reg,
                        MachineOperand::TiedDefUse(w) => w.to_reg() == reg,
                        _ => false,
                    }}) {{
                        ops_{index}.push(MachineOperand::Use(reg));
                    }}
                }}"#
                )
                .unwrap();
            }
            for r in &implicit_defs {
                let r_val = *r;
                writeln!(
                    output,
                    r#"                {{
                    let reg = Reg::new_preg({r_val});
                    if !ops_{index}.iter().any(|op| match op {{
                        MachineOperand::Def(w) => w.to_reg() == reg,
                        MachineOperand::TiedDefUse(w) => w.to_reg() == reg,
                        _ => false,
                    }}) {{
                        ops_{index}.push(MachineOperand::Def(crate::mir::Writable(reg)));
                    }}
                }}"#
                )
                .unwrap();
            }
        }

        writeln!(
            output,
            "                let inst_{} = MachineInst::build_generic(",
            index
        )
        .unwrap();
        writeln!(
            output,
            "                    MachineOpcode::Target(TargetInst::{}.as_u32()),",
            opcode
        )
        .unwrap();
        writeln!(output, "                    ops_{},", index).unwrap();
        writeln!(output, "                );").unwrap();
        if emit_to_out {
            writeln!(output, "                out.push(inst_{});", index).unwrap();
        }
    } else {
        writeln!(
            output,
            "                let mut ops_{} = SmallVec::<[MachineOperand; 4]>::new();",
            index
        )
        .unwrap();
        for arg in args {
            match arg {
                Constructor::Variable(name) => match var_map.get(name) {
                    Some(BindingSource::OperandIndex(idx)) => {
                        writeln!(
                            output,
                            "                ops_{}.push(operand_by_index(inst, {}).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand mapping failed\")))?);",
                            index, idx
                        )
                        .unwrap();
                    }
                    Some(BindingSource::SchemaValue) => {
                        writeln!(
                            output,
                            "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Schema value requires a target instruction schema\")));"
                        )
                        .unwrap();
                        return;
                    }
                    None => {
                        writeln!(
                            output,
                            "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Unknown constructor variable\")));"
                        )
                        .unwrap();
                        return;
                    }
                },
                Constructor::Imm(i) => {
                    writeln!(
                        output,
                        "                ops_{}.push(MachineOperand::Imm({}));",
                        index, i
                    )
                    .unwrap();
                }
                Constructor::Reg(name) => {
                    let enc = reg_map.get(name).copied().unwrap_or(0);
                    writeln!(
                        output,
                        "                ops_{}.push(MachineOperand::Use(Reg::new_preg({})));",
                        index, enc
                    )
                    .unwrap();
                }
                _ => {
                    writeln!(
                        output,
                        "                return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Unsupported constructor arg\")));"
                    )
                    .unwrap();
                    return;
                }
            }
        }

        writeln!(
            output,
            "                let inst_{} = MachineInst::build_generic(",
            index
        )
        .unwrap();
        writeln!(
            output,
            "                    MachineOpcode::Target(TargetInst::{}.as_u32()),",
            opcode
        )
        .unwrap();
        writeln!(output, "                    ops_{},", index).unwrap();
        writeln!(output, "                );").unwrap();
        if emit_to_out {
            writeln!(output, "                out.push(inst_{});", index).unwrap();
        }
    }
}

fn inst_def_has_def_operands(operands: &[OperandConstraint]) -> bool {
    operands.iter().any(|op| {
        matches!(
            op,
            OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
        )
    })
}

fn min_explicit_args_from(operands: &[OperandConstraint], start: usize) -> usize {
    operands[start..]
        .iter()
        .filter(|op| {
            !matches!(
                op,
                OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
            )
        })
        .count()
}

fn inst_def_def_operand_count(operands: &[OperandConstraint]) -> usize {
    operands
        .iter()
        .filter(|op| {
            matches!(
                op,
                OperandConstraint::Def(_) | OperandConstraint::TiedDef { .. }
            )
        })
        .count()
}

fn emit_source_def_operand(output: &mut String, index: usize, operand: &OperandConstraint) {
    let op_ctor = match operand {
        OperandConstraint::Def(_) => "Def",
        OperandConstraint::TiedDef { .. } => "TiedDefUse",
        _ => panic!("source defs can only satisfy def-like operands"),
    };
    writeln!(
        output,
        r#"                {{
                    let reg = *source_defs_{index}
                        .get(source_def_cursor_{index})
                        .ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from("Source def mapping failed")))?;
                    source_def_cursor_{index} += 1;
                    ops_{index}.push(MachineOperand::{op_ctor}(crate::mir::Writable(reg)));
                }}"#
    )
    .unwrap();
}

fn emit_schema_source_def_operand(
    output: &mut String,
    index: usize,
    operand: &OperandConstraint,
    schema_var: &str,
    field: &str,
) {
    let op_ctor = match operand {
        OperandConstraint::Def(_) => "Def",
        OperandConstraint::TiedDef { .. } => "TiedDefUse",
        _ => panic!("schema source defs can only satisfy def-like operands"),
    };
    writeln!(
        output,
        r#"                ops_{index}.push(MachineOperand::{op_ctor}(crate::mir::Writable(reg_value({schema_var}.{field}.clone()).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from("Schema reg mapping failed")))?)));"#
    )
    .unwrap();
}

fn infer_schema_source_def_field(args: &[PatternArg]) -> Option<&str> {
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
    match arg {
        Constructor::Variable(name) => match var_map.get(name) {
            Some(BindingSource::SchemaValue) => {
                let push = schema_value_operand_expr(name, operand);
                writeln!(output, "                ops_{index}.push({push});").unwrap();
            }
            Some(BindingSource::OperandIndex(op_index)) => {
                let push = operand_index_expr(*op_index, operand);
                writeln!(output, "                ops_{index}.push({push});").unwrap();
            }
            None => {
                panic!("unknown constructor variable {}", name);
            }
        },
        Constructor::Imm(i) => match operand {
            OperandConstraint::Imm(_) => {
                writeln!(
                    output,
                    "                ops_{index}.push(MachineOperand::Imm({i}));"
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
                    format!("MachineOperand::Use(Reg::new_preg({enc}))")
                }
                OperandConstraint::Def(_) => {
                    format!("MachineOperand::Def(crate::mir::Writable(Reg::new_preg({enc})))")
                }
                OperandConstraint::TiedDef { .. } => format!(
                    "MachineOperand::TiedDefUse(crate::mir::Writable(Reg::new_preg({enc})))"
                ),
                OperandConstraint::StackSlot(_) => {
                    panic!("physical register constructor cannot satisfy a stackslot operand")
                }
                _ => panic!("physical register constructor cannot satisfy this operand kind"),
            };
            writeln!(output, "                ops_{index}.push({push});").unwrap();
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
            "MachineOperand::Use(reg_value({rust_name}).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Schema reg mapping failed\")))?)"
        ),
        OperandConstraint::Def(_) => format!(
            "MachineOperand::Def(crate::mir::Writable(reg_value({rust_name}).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Schema reg mapping failed\")))?))"
        ),
        OperandConstraint::TiedDef { .. } => format!(
            "MachineOperand::TiedDefUse(crate::mir::Writable(reg_value({rust_name}).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Schema reg mapping failed\")))?))"
        ),
        OperandConstraint::Imm(_) => format!("MachineOperand::Imm({rust_name}.into())"),
        OperandConstraint::Block(_) => format!("MachineOperand::Block({rust_name})"),
        OperandConstraint::Global(_) => format!("MachineOperand::Global({rust_name})"),
        OperandConstraint::StackSlot(_) => format!("MachineOperand::StackSlot({rust_name})"),
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
            "MachineOperand::Use(operand_by_index(inst, {op_index}).and_then(|op| op.as_reg()).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand reg mapping failed\")))?)"
        ),
        OperandConstraint::Def(_) => format!(
            "MachineOperand::Def(crate::mir::Writable(operand_by_index(inst, {op_index}).and_then(|op| op.as_reg()).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand reg mapping failed\")))?))"
        ),
        OperandConstraint::TiedDef { .. } => format!(
            "MachineOperand::TiedDefUse(crate::mir::Writable(operand_by_index(inst, {op_index}).and_then(|op| op.as_reg()).ok_or_else(|| crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand reg mapping failed\")))?))"
        ),
        OperandConstraint::Imm(_) => format!(
            "match operand_by_index(inst, {op_index}) {{ Some(MachineOperand::Imm(v)) => MachineOperand::Imm(v), _ => return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand immediate mapping failed\"))), }}"
        ),
        OperandConstraint::Block(_) => format!(
            "match operand_by_index(inst, {op_index}) {{ Some(MachineOperand::Block(v)) => MachineOperand::Block(v), _ => return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand block mapping failed\"))), }}"
        ),
        OperandConstraint::Global(_) => format!(
            "match operand_by_index(inst, {op_index}) {{ Some(MachineOperand::Global(v)) => MachineOperand::Global(v), _ => return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand global mapping failed\"))), }}"
        ),
        OperandConstraint::StackSlot(_) => format!(
            "match operand_by_index(inst, {op_index}) {{ Some(MachineOperand::StackSlot(v)) => MachineOperand::StackSlot(v), _ => return Err(crate::error::Error::select(inst.opcode.clone(), alloc::string::String::from(\"Operand stackslot mapping failed\"))), }}"
        ),
    }
}

fn collect_reg_encs(module: &crate::ast::Module) -> HashMap<String, u32> {
    let mut map = HashMap::new();
    for def in &module.defs {
        if let Def::Reg(reg) = def {
            map.insert(reg.name.clone(), reg.hw_enc);
        }
    }
    map
}

fn collect_canonical_regs<'a>(
    module: &'a crate::ast::Module,
) -> BTreeMap<u32, &'a crate::ast::RegDef> {
    let mut regs = BTreeMap::new();
    for def in &module.defs {
        if let Def::Reg(reg) = def {
            regs.entry(reg.hw_enc).or_insert(reg);
        }
    }
    regs
}

fn collect_reserved_reg_encs(module: &crate::ast::Module) -> BTreeSet<u32> {
    let mut regs = BTreeSet::new();
    for def in &module.defs {
        if let Def::Reg(reg) = def {
            if reg.reserved {
                regs.insert(reg.hw_enc);
            }
        }
    }
    regs
}

fn collect_special_role_regs<'a>(
    module: &'a crate::ast::Module,
) -> BTreeMap<String, &'a crate::ast::RegDef> {
    let mut roles = BTreeMap::new();
    let canonical_regs = collect_canonical_regs(module);
    for reg in canonical_regs.values() {
        for role in &reg.roles {
            roles.entry(role.clone()).or_insert(*reg);
        }
    }
    roles
}

fn generate_register_descriptors(output: &mut String, module: &crate::ast::Module) {
    let regs: Vec<_> = module
        .defs
        .iter()
        .filter_map(|def| match def {
            Def::Reg(reg) => Some(reg),
            _ => None,
        })
        .collect();
    let reg_classes: Vec<_> = module
        .defs
        .iter()
        .filter_map(|def| match def {
            Def::RegClass(class) => Some(class),
            _ => None,
        })
        .collect();

    if regs.is_empty() && reg_classes.is_empty() {
        return;
    }

    writeln!(output, "\n/// Register constants generated from `def-reg`.").unwrap();
    for reg in regs {
        let const_name = sanitize_ident(&format!("REG_{}", reg.name));
        writeln!(
            output,
            "pub const {}: Reg = Reg({});",
            const_name, reg.hw_enc
        )
        .unwrap();
    }

    let canonical_regs = collect_canonical_regs(module);
    let reg_encs = collect_reg_encs(module);
    let reserved_encs = collect_reserved_reg_encs(module);
    let special_role_regs = collect_special_role_regs(module);
    writeln!(
        output,
        "\n/// Canonical physical register descriptors generated from `def-reg`."
    )
    .unwrap();
    writeln!(output, "pub const PHYS_REG_INFOS: &[RegInfo] = &[").unwrap();
    for reg in canonical_regs.values() {
        let const_name = sanitize_ident(&format!("REG_{}", reg.name));
        writeln!(
            output,
            "    RegInfo {{ preg: {const_name}, name: \"{name}\", size: {size}, hw_encoding: {enc} }},",
            const_name = const_name,
            name = reg.name.to_ascii_lowercase(),
            size = reg.size,
            enc = reg.hw_enc,
        )
        .unwrap();
    }
    writeln!(output, "];").unwrap();

    writeln!(
        output,
        "\n/// Reserved physical registers generated from `def-reg`."
    )
    .unwrap();
    let reserved_regs = canonical_regs
        .values()
        .filter(|reg| reserved_encs.contains(&reg.hw_enc))
        .map(|reg| sanitize_ident(&format!("REG_{}", reg.name)))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "pub const RESERVED_REGS: &[Reg] = &[{}];",
        reserved_regs
    )
    .unwrap();

    if !special_role_regs.is_empty() {
        writeln!(
            output,
            "\n/// Special-register roles generated from `def-reg`."
        )
        .unwrap();
        for (role, reg) in special_role_regs {
            let const_name = sanitize_ident(&format!("SPECIAL_REG_{}", role)).to_ascii_uppercase();
            let reg_name = sanitize_ident(&format!("REG_{}", reg.name));
            writeln!(output, "pub const {}: Reg = {};", const_name, reg_name).unwrap();
        }
    }

    if reg_classes.is_empty() {
        return;
    }

    writeln!(
        output,
        "\n/// Register-class member slices generated from `def-regclass`."
    )
    .unwrap();
    for class in reg_classes {
        let const_name = sanitize_ident(&format!("REGCLASS_{}", class.name));
        let regs = class
            .regs
            .iter()
            .map(|reg| sanitize_ident(&format!("REG_{}", reg)))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "pub const {}: &[Reg] = &[{}];", const_name, regs).unwrap();

        let allocatable_name = format!("{}_ALLOCATABLE", const_name);
        let allocatable = class
            .regs
            .iter()
            .filter_map(|reg_name| {
                let reg_enc = reg_encs.get(reg_name).copied()?;
                (!reserved_encs.contains(&reg_enc))
                    .then(|| sanitize_ident(&format!("REG_{}", reg_name)))
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "pub const {}: &[Reg] = &[{}];",
            allocatable_name, allocatable
        )
        .unwrap();
    }
}

fn generate_abi_descriptors(
    output: &mut String,
    module: &crate::ast::Module,
) -> Result<(), String> {
    let reg_map = collect_reg_encs(module);
    let abis: Vec<_> = module
        .defs
        .iter()
        .filter_map(|def| match def {
            Def::Abi(abi) => Some(abi),
            _ => None,
        })
        .collect();

    if abis.is_empty() {
        return Ok(());
    }

    writeln!(output, "\n/// ABI descriptors generated from `def-abi`.").unwrap();

    for abi in abis {
        let prefix = sanitize_ident(&format!("ABI_{}", abi.name));
        let args_name = format!("{}_ARGS", prefix);
        let returns_name = format!("{}_RETURNS", prefix);
        let preserved_name = format!("{}_PRESERVED", prefix);

        generate_abi_pool_array(output, &args_name, &abi.args, &reg_map);
        generate_abi_pool_array(output, &returns_name, &abi.returns, &reg_map);
        generate_abi_preserved_array(output, &preserved_name, &abi.preserved, &reg_map);

        let arch = abi_arch_expr(&abi.arch)?;
        let align = abi.stack.align.unwrap_or(16);
        let (incoming_base_reg, incoming_base_offset) = abi
            .stack
            .incoming_base
            .as_ref()
            .map(|(reg, off)| {
                (
                    format!("Some({})", reg_expr(reg_map.get(reg).copied())),
                    *off,
                )
            })
            .unwrap_or(("None".to_string(), 0));
        let (outgoing_slot_size, outgoing_slot_align) = abi.stack.outgoing_slot.unwrap_or((8, 8));
        let classifier = abi
            .classifier
            .as_ref()
            .map(|name| format!("Some(\"{}\")", name))
            .unwrap_or_else(|| "None".to_string());

        writeln!(
            output,
            r#"
pub static {prefix}: AbiDescriptor = AbiDescriptor {{
    name: "{name}",
    arch: {arch},
    classifier: {classifier},
    stack: AbiStackDescriptor {{
        align: {align},
        incoming_base_reg: {incoming_base_reg},
        incoming_base_offset: {incoming_base_offset},
        outgoing_slot_size: {outgoing_slot_size},
        outgoing_slot_align: {outgoing_slot_align},
    }},
    args: {args_name},
    returns: {returns_name},
    preserved: {preserved_name},
}};
"#,
            prefix = prefix,
            name = abi.name,
            arch = arch,
            classifier = classifier,
            align = align,
            incoming_base_reg = incoming_base_reg,
            incoming_base_offset = incoming_base_offset,
            outgoing_slot_size = outgoing_slot_size,
            outgoing_slot_align = outgoing_slot_align,
            args_name = args_name,
            returns_name = returns_name,
            preserved_name = preserved_name,
        )
        .unwrap();
    }
    Ok(())
}

fn generate_abi_pool_array(
    output: &mut String,
    const_name: &str,
    classes: &[crate::ast::AbiClassRegsDef],
    reg_map: &HashMap<String, u32>,
) {
    writeln!(output, "pub const {}: &[AbiRegisterPool] = &[", const_name).unwrap();
    for class in classes {
        let regs = class
            .regs
            .iter()
            .map(|reg| reg_expr(reg_map.get(reg).copied()))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "    AbiRegisterPool {{ class: {}, regs: &[{}] }},",
            abi_value_class_expr(&class.class),
            regs
        )
        .unwrap();
    }
    writeln!(output, "];").unwrap();
}

fn generate_abi_preserved_array(
    output: &mut String,
    const_name: &str,
    preserved: &[crate::ast::AbiPreservedSetDef],
    reg_map: &HashMap<String, u32>,
) {
    writeln!(output, "pub const {}: &[AbiPreservedSet] = &[", const_name).unwrap();
    for set in preserved {
        let regs = set
            .regs
            .iter()
            .map(|reg| reg_expr(reg_map.get(reg).copied()))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "    AbiPreservedSet {{ bank: \"{}\", regs: &[{}] }},",
            set.bank, regs
        )
        .unwrap();
    }
    writeln!(output, "];").unwrap();
}

fn abi_value_class_expr(class: &str) -> &'static str {
    match class {
        "Integer" | "Int" => "AbiValueClass::Integer",
        "Float" => "AbiValueClass::Float",
        "Vector" => "AbiValueClass::Vector",
        "Memory" => "AbiValueClass::Memory",
        _ => "AbiValueClass::Integer",
    }
}

fn abi_arch_expr(arch: &str) -> Result<&'static str, String> {
    match arch {
        "X86_64" => Ok("TargetArch::X86_64"),
        "AArch64" => Ok("TargetArch::AArch64"),
        "Riscv64" => Ok("TargetArch::Riscv64"),
        "Wasm32" => Ok("TargetArch::Wasm32"),
        "Wasm64" => Ok("TargetArch::Wasm64"),
        _ => Err(format!("unsupported ABI architecture `{}`", arch)),
    }
}

fn reg_expr(enc: Option<u32>) -> String {
    match enc {
        Some(enc) => format!("Reg({})", enc),
        None => "Reg(0)".to_string(),
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn preprocess_isle(input: &str) -> Result<String, String> {
    let lines: Vec<String> = input.lines().map(|line| line.to_string()).collect();
    let (expanded, idx) = expand_lines(&lines, 0, &HashMap::new())?;
    debug_assert_eq!(idx, lines.len());
    Ok(expanded)
}

fn expand_lines(
    lines: &[String],
    mut idx: usize,
    vars: &HashMap<String, String>,
) -> Result<(String, usize), String> {
    let mut out = String::new();
    while idx < lines.len() {
        let line = &lines[idx];
        let trimmed = line.trim_start();
        if trimmed == "@end" {
            return Ok((out, idx + 1));
        }

        if let Some(spec) = trimmed.strip_prefix("@for ") {
            let (names, inline_values) = parse_for_header(spec)?;
            if let Some(values) = inline_values {
                let body_start = idx + 1;
                let body_end = find_matching_end(lines, body_start)?;
                let body = lines[body_start..body_end].join("\n");
                for tuple in values {
                    let mut scoped = vars.clone();
                    for (name, value) in names.iter().zip(tuple.iter()) {
                        scoped.insert(name.clone(), value.clone());
                    }
                    let substituted = substitute_vars(&body, &scoped);
                    let (expanded, _) = expand_lines(
                        &substituted
                            .lines()
                            .map(|line| line.to_string())
                            .collect::<Vec<_>>(),
                        0,
                        &HashMap::new(),
                    )?;
                    out.push_str(&expanded);
                    if !expanded.ends_with('\n') {
                        out.push('\n');
                    }
                }
                idx = body_end + 1;
            } else {
                let body_start = idx + 1;
                let do_idx = find_for_do(lines, body_start)?;
                let body_end = find_matching_end(lines, do_idx + 1)?;
                let cases = parse_for_cases(&lines[body_start..do_idx], names.len())?;
                let body = lines[do_idx + 1..body_end].join("\n");
                for tuple in cases {
                    let mut scoped = vars.clone();
                    for (name, value) in names.iter().zip(tuple.iter()) {
                        scoped.insert(name.clone(), value.clone());
                    }
                    let substituted = substitute_vars(&body, &scoped);
                    let (expanded, _) = expand_lines(
                        &substituted
                            .lines()
                            .map(|line| line.to_string())
                            .collect::<Vec<_>>(),
                        0,
                        &HashMap::new(),
                    )?;
                    out.push_str(&expanded);
                    if !expanded.ends_with('\n') {
                        out.push('\n');
                    }
                }
                idx = body_end + 1;
            }
            continue;
        }

        out.push_str(&substitute_vars(line, vars));
        out.push('\n');
        idx += 1;
    }
    Ok((out, idx))
}

fn find_matching_end(lines: &[String], mut idx: usize) -> Result<usize, String> {
    let mut depth = 0usize;
    while idx < lines.len() {
        let trimmed = lines[idx].trim_start();
        if trimmed.strip_prefix("@for ").is_some() {
            depth += 1;
        } else if trimmed == "@end" {
            if depth == 0 {
                return Ok(idx);
            }
            depth -= 1;
        }
        idx += 1;
    }
    Err("missing @end for @for block".to_string())
}

fn parse_for_names(spec: &str) -> Result<Vec<String>, String> {
    let names: Vec<String> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if names.is_empty() {
        Err(format!("invalid @for header: missing loop variables in {spec}"))
    } else {
        Ok(names)
    }
}

fn parse_for_header(spec: &str) -> Result<(Vec<String>, Option<Vec<Vec<String>>>), String> {
    let Some((lhs, rhs)) = spec.split_once(" in") else {
        return Ok((parse_for_names(spec)?, None));
    };

    let names = parse_for_names(lhs)?;

    let rhs = rhs.trim();
    if rhs.is_empty() {
        return Ok((names, None));
    }

    let values = rhs
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|tuple| parse_for_tuple(tuple, names.len()))
        .collect::<Result<Vec<_>, _>>()?;

    if values.is_empty() {
        return Err(format!("invalid @for header: no tuples found in {spec}"));
    }

    Ok((names, Some(values)))
}

fn parse_for_cases(lines: &[String], arity: usize) -> Result<Vec<Vec<String>>, String> {
    let mut values = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(";;") {
            continue;
        }
        values.push(parse_for_tuple(trimmed, arity)?);
    }
    if values.is_empty() {
        return Err("invalid @for block: no cases found before @do".to_string());
    }
    Ok(values)
}

fn parse_for_tuple(tuple: &str, arity: usize) -> Result<Vec<String>, String> {
    let vals: Vec<String> = tuple
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if vals.len() != arity {
        Err(format!(
            "invalid @for tuple `{tuple}`: expected {} values, found {}",
            arity,
            vals.len()
        ))
    } else {
        Ok(vals)
    }
}

fn find_for_do(lines: &[String], mut idx: usize) -> Result<usize, String> {
    let mut depth = 0usize;
    while idx < lines.len() {
        let trimmed = lines[idx].trim_start();
        if trimmed.strip_prefix("@for ").is_some() {
            depth += 1;
        } else if trimmed == "@end" {
            if depth == 0 {
                return Err("missing @do for @for block".to_string());
            }
            depth -= 1;
        } else if trimmed == "@do" && depth == 0 {
            return Ok(idx);
        }
        idx += 1;
    }
    Err("missing @do for @for block".to_string())
}

fn substitute_vars(text: &str, vars: &HashMap<String, String>) -> String {
    let mut out = text.to_string();
    for (name, value) in vars {
        let needle = format!("{{{{{}}}}}", name);
        out = out.replace(&needle, value);
    }
    out
}

// =============================================================================
// 主编译函数
// =============================================================================

pub fn compile(input: &str, arch: &str) -> Result<String, String> {
    let input = preprocess_isle(input)?;

    // 解析输入
    let module = parse_input(&input)?;

    // 收集定义
    let (extractors, templates, macros) = collect_definitions(&module);

    // 实例化模板
    let final_inst_defs = instantiate_templates(&module, &templates);
    let needs_positional_helpers = module_has_positional_rules(&module);
    let needs_source_defs_helper = module_needs_source_defs_helper(&module, &final_inst_defs);

    // 生成输出
    let mut output = String::new();
    generate_header(
        &mut output,
        arch,
        needs_positional_helpers,
        needs_source_defs_helper,
    );
    generate_register_descriptors(&mut output, &module);
    generate_cpu_info(&mut output, &module);
    generate_abi_descriptors(&mut output, &module)?;
    generate_enum(&mut output, &final_inst_defs);
    generate_enum_conversions(&mut output, &final_inst_defs);
    generate_target_inst_metadata(&mut output, &module, &final_inst_defs);
    generate_generic_inst_metadata(&mut output, &module, &final_inst_defs);
    generate_emit_method(&mut output, &final_inst_defs, &macros);
    generate_pre_isel_rewrite_tables(&mut output, &module);
    generate_select_instruction(&mut output, &module, &extractors, &final_inst_defs, arch);

    Ok(output)
}
