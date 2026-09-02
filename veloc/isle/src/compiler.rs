mod generate;
mod preprocess;
mod select;

use crate::ast::Def;
use crate::{EmitExpr, Expr, ExtractorDef, MacroDef, OperandConstraint, parser};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct FinalInstDef {
    operands: Vec<OperandConstraint>,
    implicit_uses: Vec<String>,
    implicit_defs: Vec<String>,
    clobbers: Vec<String>,
    emit: Vec<EmitExpr>,
    is_pseudo: bool,
}

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

pub fn compile(input: &str, arch: &str) -> Result<String, String> {
    let input = preprocess::preprocess_isle(input)?;
    let module = parse_input(&input)?;
    let (extractors, templates, macros) = collect_definitions(&module);
    let final_inst_defs = instantiate_templates(&module, &templates);
    let needs_positional_helpers = select::module_has_positional_rules(&module);
    let needs_source_defs_helper =
        select::module_needs_source_defs_helper(&module, &final_inst_defs);

    let mut output = String::new();
    generate::generate_header(
        &mut output,
        arch,
        needs_positional_helpers,
        needs_source_defs_helper,
    );
    generate::generate_register_descriptors(&mut output, &module);
    generate::generate_cpu_info(&mut output, &module);
    generate::generate_abi_descriptors(&mut output, &module)?;
    generate::generate_enum(&mut output, &final_inst_defs);
    generate::generate_enum_conversions(&mut output, &final_inst_defs);
    generate::generate_target_inst_metadata(&mut output, &module, &final_inst_defs);
    select::generate_generic_inst_metadata(&mut output, &module, &final_inst_defs);
    generate::generate_emit_method(&mut output, &final_inst_defs, &macros);
    select::generate_select_instruction(&mut output, &module, &extractors, &final_inst_defs, arch);

    Ok(output)
}
