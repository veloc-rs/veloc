//! Constant Folding Pass
//!
//! This pass folds constant expressions into their results.
//! For example: `iconst 1 + iconst 2` -> `iconst 3`

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use veloc_analyzer::AnalysisManager;
use veloc_ir::constant::Constant;
use veloc_ir::function::Function;
use veloc_ir::inst::{Inst, InstructionData};

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
                    let inst_data = InstructionData::from(folded_const);

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
            lhs.binary_op(rhs, *opcode)
        }
        InstructionData::Unary { opcode, arg } => {
            let val = func.dfg.as_const(*arg)?;
            val.unary_op(*opcode)
        }
        InstructionData::IntCompare { kind, args } => {
            let lhs = func.dfg.as_const(args[0])?;
            let rhs = func.dfg.as_const(args[1])?;
            lhs.icmp(rhs, *kind)
        }
        _ => None,
    }
}
