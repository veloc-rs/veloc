//! Constant Folding Pass
//!
//! This pass folds constant expressions into their results.
//! For example: `iconst 1 + iconst 2` -> `iconst 3`

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use veloc_analyzer::AnalysisManager;
use veloc_ir::constant::Constant;
use veloc_ir::function::Function;
use veloc_ir::inst::{Inst, InstructionData};
use veloc_ir::{IntCC, Opcode};

const CONSTANT_FOLDING: &str = "constant_folding";

pub struct ConstantFoldingPass;

impl FunctionPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "ConstantFoldingPass"
    }

    fn run(
        &self,
        func: &mut Function,
        am: &mut AnalysisManager,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        let changed =
            run_constant_folding(func, am, config.is_debug_enabled(CONSTANT_FOLDING), metrics);
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_constant_folding(
    func: &mut Function,
    am: &mut AnalysisManager,
    print_folded: bool,
    metrics: &mut Metrics,
) -> bool {
    let mut changed = false;
    let mut fold_count = 0u64;

    loop {
        let mut pass_changed = false;

        // 获取指令快照进行遍历。
        let insts: Vec<Inst> = func
            .layout
            .block_order
            .iter()
            .flat_map(|&block| func.layout.blocks[block].insts.iter().copied())
            .collect();

        for inst in insts {
            // 检查指令是否有效（之前的 fold 可能已经将其变为 Nop）
            if matches!(func.dfg.instructions[inst], InstructionData::Nop) {
                continue;
            }

            if let Some(folded_const) = try_fold_instruction(func, inst) {
                if print_folded {
                    log::info!(
                        "[ConstantFolding] Folding instruction {} results in {:?}",
                        inst,
                        folded_const
                    );
                }

                let results = func.dfg.inst_results(inst);
                if results.len() == 1 {
                    // 指令变形 (Instruction Morphing)：保持相同的 Value ID，只需更改其定义逻辑。
                    let inst_data = match folded_const {
                        Constant::I8(v) => InstructionData::Iconst { value: v as u64 },
                        Constant::I16(v) => InstructionData::Iconst { value: v as u64 },
                        Constant::I32(v) => InstructionData::Iconst { value: v as u64 },
                        Constant::I64(v) => InstructionData::Iconst { value: v as u64 },
                        Constant::F32(v) => InstructionData::Fconst {
                            value: v.to_bits() as u64,
                        },
                        Constant::F64(v) => InstructionData::Fconst { value: v.to_bits() },
                        Constant::Bool(v) => InstructionData::Bconst { value: v },
                    };

                    // 在更改指令之前，从使用-定义链中脱离原操作数
                    let use_def = am.use_def_mut(func);
                    use_def.detach_inst(func, inst);

                    // 替换指令数据
                    func.dfg.replace_inst(inst, inst_data);

                    // 注意：无需 RAUW，因为原结果 Value 仍然由该指令定义。
                    func.bump_revision();
                    am.update_use_def_revision(func);

                    pass_changed = true;
                    fold_count += 1;
                }
            }
        }

        if !pass_changed {
            break;
        }
        changed = true;
    }

    if changed {
        metrics.add("constant_folding.folded_insts", fold_count);
        compact_layout(func);
    }

    changed
}

/// 从布局中移除 Nop 指令
fn compact_layout(func: &mut Function) {
    for block in &func.layout.block_order.clone() {
        func.layout.blocks[*block]
            .insts
            .retain(|&inst| !matches!(func.dfg.instructions[inst], InstructionData::Nop));
    }
}

/// 尝试折叠单个指令
fn try_fold_instruction(func: &Function, inst: Inst) -> Option<Constant> {
    let idata = &func.dfg.instructions[inst];
    let results = func.dfg.inst_results(inst);

    if results.len() != 1 {
        return None;
    }

    match idata {
        InstructionData::Binary { opcode, args } => {
            let lhs = func.dfg.as_const(args[0])?;
            let rhs = func.dfg.as_const(args[1])?;
            fold_binary_op(*opcode, lhs, rhs)
        }
        InstructionData::Unary { opcode, arg } => {
            let val = func.dfg.as_const(*arg)?;
            fold_unary_op(*opcode, val)
        }
        InstructionData::IntCompare { kind, args } => {
            let lhs = func.dfg.as_const(args[0])?;
            let rhs = func.dfg.as_const(args[1])?;
            fold_icmp(kind, lhs, rhs)
        }
        _ => None,
    }
}

/// 折叠二元运算
fn fold_binary_op(opcode: Opcode, lhs: Constant, rhs: Constant) -> Option<Constant> {
    let l_val = lhs.as_i64()?;
    let r_val = rhs.as_i64()?;

    let (result, is_i64) = match (lhs, rhs) {
        (Constant::I64(_), Constant::I64(_)) => {
            (perform_binary_fold(opcode, l_val, r_val, 64)?, true)
        }
        (Constant::I32(_), Constant::I32(_)) => {
            (perform_binary_fold(opcode, l_val, r_val, 32)?, false)
        }
        _ => return None,
    };

    if is_i64 {
        Some(Constant::I64(result))
    } else {
        Some(Constant::I32(result as i32))
    }
}

fn perform_binary_fold(opcode: Opcode, lhs: i64, rhs: i64, bits: u32) -> Option<i64> {
    let result = match opcode {
        Opcode::IAdd => lhs.wrapping_add(rhs),
        Opcode::ISub => lhs.wrapping_sub(rhs),
        Opcode::IMul => lhs.wrapping_mul(rhs),
        Opcode::IDivS => {
            if rhs == 0 || (lhs == i64::MIN && rhs == -1) {
                return None;
            }
            lhs.wrapping_div(rhs)
        }
        Opcode::IDivU => {
            if rhs == 0 {
                return None;
            }
            let l = truncate_u64(lhs as u64, bits);
            let r = truncate_u64(rhs as u64, bits);
            l.wrapping_div(r) as i64
        }
        Opcode::IAnd => lhs & rhs,
        Opcode::IOr => lhs | rhs,
        Opcode::IXor => lhs ^ rhs,
        Opcode::IShl => lhs.wrapping_shl((rhs as u32) % bits),
        Opcode::IShrS => {
            if bits == 32 {
                (lhs as i32).wrapping_shr((rhs as u32) % 32) as i64
            } else {
                lhs.wrapping_shr((rhs as u32) % 64)
            }
        }
        Opcode::IShrU => {
            let l = truncate_u64(lhs as u64, bits);
            l.wrapping_shr((rhs as u32) % bits) as i64
        }
        _ => return None,
    };

    Some(truncate_i64(result, bits))
}

/// 折叠一元运算
fn fold_unary_op(opcode: Opcode, val: Constant) -> Option<Constant> {
    let v = val.as_i64()?;
    let bits = match val {
        Constant::I64(_) => 64,
        Constant::I32(_) => 32,
        _ => return None,
    };

    let result = match opcode {
        Opcode::INeg => v.wrapping_neg(),
        Opcode::IClz => {
            let v_u = truncate_u64(v as u64, bits);
            v_u.leading_zeros().saturating_sub(64 - bits) as i64
        }
        Opcode::ICtz => {
            let v_u = truncate_u64(v as u64, bits);
            v_u.trailing_zeros().min(bits) as i64
        }
        Opcode::IPopcnt => {
            let v_u = truncate_u64(v as u64, bits);
            v_u.count_ones() as i64
        }
        Opcode::IEqz => return Some(Constant::Bool(v == 0)),
        _ => return None,
    };

    let truncated = truncate_i64(result, bits);
    if bits == 64 {
        Some(Constant::I64(truncated))
    } else {
        Some(Constant::I32(truncated as i32))
    }
}

/// 折叠整数比较
fn fold_icmp(kind: &IntCC, lhs: Constant, rhs: Constant) -> Option<Constant> {
    let l = lhs.as_i64()?;
    let r = rhs.as_i64()?;

    let (bits, unsigned) = match (lhs, rhs) {
        (Constant::I64(_), Constant::I64(_)) => (64, kind.is_unsigned()),
        (Constant::I32(_), Constant::I32(_)) => (32, kind.is_unsigned()),
        _ => return None,
    };

    let result = if unsigned {
        let lu = truncate_u64(l as u64, bits);
        let ru = truncate_u64(r as u64, bits);
        match kind {
            IntCC::Eq => lu == ru,
            IntCC::Ne => lu != ru,
            IntCC::LtU => lu < ru,
            IntCC::LeU => lu <= ru,
            IntCC::GtU => lu > ru,
            IntCC::GeU => lu >= ru,
            _ => unreachable!(),
        }
    } else {
        let ls = truncate_i64(l, bits);
        let rs = truncate_i64(r, bits);
        match kind {
            IntCC::Eq => ls == rs,
            IntCC::Ne => ls != rs,
            IntCC::LtS => ls < rs,
            IntCC::LeS => ls <= rs,
            IntCC::GtS => ls > rs,
            IntCC::GeS => ls >= rs,
            _ => unreachable!(),
        }
    };

    Some(Constant::Bool(result))
}

fn truncate_u64(val: u64, bits: u32) -> u64 {
    if bits == 64 {
        val
    } else {
        val & ((1u64 << bits) - 1)
    }
}

fn truncate_i64(val: i64, bits: u32) -> i64 {
    if bits == 64 {
        val
    } else {
        let mask = (1u64 << bits) - 1;
        let truncated = (val as u64) & mask;
        let sign_bit = 1u64 << (bits - 1);
        if truncated & sign_bit != 0 {
            (truncated | !mask) as i64
        } else {
            truncated as i64
        }
    }
}
