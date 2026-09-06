use crate::ast::{Def, OperandConstraint};
use crate::{EmitExpr, Expr, MacroDef};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

use super::{FinalInstDef, subst_expr};

/// 收集寄存器硬件编码
pub(crate) fn collect_reg_encs(module: &crate::ast::Module) -> HashMap<String, u32> {
    let mut map = HashMap::new();
    for def in &module.defs {
        if let Def::Reg(reg) = def {
            map.insert(reg.name.clone(), reg.hw_enc);
        }
    }
    map
}

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
            format!(
                "(match &inst.operands[{}] {{ {}, _ => return Err(crate::error::Error::emit(inst.opcode.clone(), alloc::format!(\"Operand type mismatch at index {} for {{}}\", \"{}\"))) }})",
                index, arm, index, var_name
            )
        }
        None => "0".to_string(),
    }
}

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
                OperandConstraint::Imm(_) => "MachineOperand::Imm(val) => *val as u8",
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

/// Keep literal information while expanding encoding macros, so constant
/// bit operations are folded before emitting Rust instead of reparsing strings.
enum EncodingExpr {
    Constant(i64),
    Code(String),
}

#[cfg(test)]
mod encoding_tests {
    use super::*;

    #[test]
    fn folds_literal_bit_operations_after_macro_expansion() {
        let mut macros = HashMap::new();
        macros.insert(
            "mask".into(),
            MacroDef {
                name: "mask".into(),
                args: vec!["x".into()],
                body: Expr::BitAnd(Box::new(Expr::Variable("x".into())), Box::new(Expr::Int(7))),
            },
        );
        let expr = Expr::Shl(
            Box::new(Expr::Call("mask".into(), vec![Expr::Int(7)])),
            Box::new(Expr::Int(3)),
        );
        assert!(matches!(
            generate_expr(&expr, &[], &macros),
            EncodingExpr::Constant(56)
        ));
        let expr = Expr::Call("bit-and".into(), vec![Expr::Int(0), Expr::Int(7)]);
        assert!(matches!(
            generate_expr(&expr, &[], &macros),
            EncodingExpr::Constant(0)
        ));
    }

    #[test]
    fn folding_does_not_discard_dynamic_evaluation_or_guess_shift_semantics() {
        let macros = HashMap::new();
        let dynamic = Expr::BitAnd(
            Box::new(Expr::Int(0)),
            Box::new(Expr::Call("read-next".into(), vec![])),
        );
        assert_eq!(
            generate_expr(&dynamic, &[], &macros).to_string(),
            "(0 & ctx.read_next())"
        );
        for (lhs, rhs) in [(-1, 3), (1, 31), (1, 64)] {
            let expr = Expr::Shl(Box::new(Expr::Int(lhs)), Box::new(Expr::Int(rhs)));
            assert!(matches!(
                generate_expr(&expr, &[], &macros),
                EncodingExpr::Code(_)
            ));
        }
    }
}

impl std::fmt::Display for EncodingExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => write!(f, "{value}"),
            Self::Code(code) => f.write_str(code),
        }
    }
}

fn binary_expr(op: &str, lhs: EncodingExpr, rhs: EncodingExpr) -> EncodingExpr {
    if let (EncodingExpr::Constant(a), EncodingExpr::Constant(b)) = (&lhs, &rhs) {
        // Unsuffixed Rust literals inherit their context. Stay within the
        // nonnegative i32 range and leave negative/overflowing shifts to Rust.
        if (0..=i64::from(i32::MAX)).contains(a) && (0..=i64::from(i32::MAX)).contains(b) {
            let value = match op {
                "|" => Some(a | b),
                "&" => Some(a & b),
                "<<" if *b < 32 => a.checked_shl(*b as u32),
                ">>" if *b < 32 => a.checked_shr(*b as u32),
                "<<" | ">>" => None,
                _ => unreachable!("known encoding bit operator"),
            };
            if let Some(value) = value.filter(|value| *value <= i64::from(i32::MAX)) {
                return EncodingExpr::Constant(value);
            }
        }
    }
    EncodingExpr::Code(format!("({lhs} {op} {rhs})"))
}

fn generate_expr(
    expr: &Expr,
    operands: &[OperandConstraint],
    macros: &HashMap<String, MacroDef>,
) -> EncodingExpr {
    let binary = |op, a, b| {
        binary_expr(
            op,
            generate_expr(a, operands, macros),
            generate_expr(b, operands, macros),
        )
    };
    match expr {
        Expr::Int(i) => EncodingExpr::Constant(*i),
        Expr::Variable(v) => EncodingExpr::Code(generate_variable(v, operands)),
        Expr::HwEnc(v) => EncodingExpr::Code(generate_hw_enc(v, operands)),
        Expr::SlotBaseHwEnc(v) => {
            EncodingExpr::Code(generate_stack_slot_expr(v, operands, "base_hw_enc"))
        }
        Expr::SlotOffset(v) => EncodingExpr::Code(generate_stack_slot_expr(v, operands, "offset")),
        Expr::SlotSize(v) => EncodingExpr::Code(generate_stack_slot_expr(v, operands, "size")),
        Expr::SlotAlign(v) => EncodingExpr::Code(generate_stack_slot_expr(v, operands, "align")),
        Expr::BitOr(a, b) => binary("|", a, b),
        Expr::BitAnd(a, b) => binary("&", a, b),
        Expr::Shl(a, b) => binary("<<", a, b),
        Expr::Shr(a, b) => binary(">>", a, b),
        Expr::Call(name, args) => {
            if let Some(m) = macros.get(name) {
                let args_map: HashMap<_, _> = m
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| args.get(i).map(|val| (name.clone(), val.clone())))
                    .collect();
                return generate_expr(&subst_expr(&m.body, &args_map), operands, macros);
            }
            if let [a, b] = args.as_slice() {
                let op = match name.as_str() {
                    "bit-or" => Some("|"),
                    "bit-and" => Some("&"),
                    "shl" => Some("<<"),
                    "shr" => Some(">>"),
                    _ => None,
                };
                if let Some(op) = op {
                    return binary(op, a, b);
                }
            }
            let args = args
                .iter()
                .map(|a| generate_expr(a, operands, macros).to_string())
                .collect::<Vec<_>>()
                .join(", ");
            EncodingExpr::Code(format!("ctx.{}({args})", name.replace("-", "_")))
        }
    }
}

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
            if !else_p.is_empty() {
                s.push_str("            } else {\n");
                for e in else_p {
                    s.push_str(&format!(
                        "                {}\n",
                        generate_emit_expr(e, operands, macros)
                    ));
                }
            }
            s.push_str("            }");
            s
        }
    }
}

pub(crate) fn generate_header(
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
        r#"use veloc_lir::{{MachineInst, MachineOperand, Reg}};
use smallvec::SmallVec;
use crate::target::arch::{{
    AbiDescriptor, AbiPreservedSet, AbiRegisterPool, AbiStackDescriptor, AbiValueClass,
    CpuDescription, FixedUseConstraint, GenericInstMetadata, LoweringContext, RegInfo,
    SelectResult, TargetArch, TargetInstMetadata, TargetTiedOperandMetadata, TiedOperandConstraint,
}};
pub use veloc_mir::Type;

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

fn reg_value_to_vreg<R: IntoOptReg>(value: R) -> Option<veloc_lir::VReg> {{
    value
        .into_opt_reg()
        .and_then(|reg| reg.is_vreg().then(|| veloc_lir::VReg::from_u32(reg.index())))
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

fn vreg_by_index(inst: &MachineInst, index: usize) -> Option<veloc_lir::VReg> {{
    let reg = inst.operands.get(index)?.as_reg()?;
    reg.is_vreg().then(|| veloc_lir::VReg::from_u32(reg.index()))
}}
"#
        )
        .unwrap();
    }
    writeln!(output).unwrap();
}

pub(crate) fn generate_enum(output: &mut String, final_inst_defs: &HashMap<String, FinalInstDef>) {
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

pub(crate) fn generate_enum_conversions(
    output: &mut String,
    final_inst_defs: &HashMap<String, FinalInstDef>,
) {
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

pub(crate) fn reg_const_name(name: &str) -> String {
    format!("REG_{}", sanitize_ident(name).to_ascii_uppercase())
}

pub(crate) fn format_slice(entries: Vec<String>) -> String {
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

pub(crate) fn generate_target_inst_metadata(
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

pub(crate) fn generate_emit_method(
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
        mfunc: &veloc_lir::MachineFunction<veloc_lir::stages::PrologueEpilogueInserted>,
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

pub(crate) fn generate_cpu_info(output: &mut String, module: &crate::ast::Module) {
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

pub(crate) fn generate_register_descriptors(output: &mut String, module: &crate::ast::Module) {
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

pub(crate) fn generate_abi_descriptors(
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
        let prefix = sanitize_ident(&format!("ABI_{}", abi.name)).to_ascii_uppercase();
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

pub(crate) fn sanitize_ident(name: &str) -> String {
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
