use crate::{AnalysisManager, FunctionPass, OptimizationConfig, PreservedAnalyses};
use hashbrown::HashSet;
use veloc_ir::function::Function;
use veloc_ir::inst::{Inst, InstructionData};
use veloc_ir::types::ValueDef;

pub struct DcePass;

impl FunctionPass for DcePass {
    fn name(&self) -> &str {
        "DcePass"
    }

    fn run(
        &self,
        func: &mut Function,
        am: &mut AnalysisManager,
        config: &OptimizationConfig,
    ) -> PreservedAnalyses {
        let changed = run_dce(func, am, config.print_removed_insts);
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_dce(func: &mut Function, am: &mut AnalysisManager, print_removed: bool) -> bool {
    let mut live_insts = HashSet::new();
    let mut worklist: Vec<Inst> = Vec::new();

    // 1. Identify roots: instructions with side effects
    for block in &func.layout.block_order {
        for &inst in &func.layout.blocks[*block].insts {
            if func.dfg.instructions[inst].has_side_effects() {
                if live_insts.insert(inst) {
                    worklist.push(inst);
                }
            }
        }
    }

    // 2. Propagate liveness back through use-def chains
    while let Some(inst) = worklist.pop() {
        inst.visit_operands(&func.dfg, |val| {
            if let ValueDef::Inst(def_inst) = func.dfg.values[val].def {
                if live_insts.insert(def_inst) {
                    worklist.push(def_inst);
                }
            }
        });
    }

    let mut changed = false;
    // 3. Mark dead instructions as Nop
    for block in &func.layout.block_order {
        for &inst in &func.layout.blocks[*block].insts {
            if !live_insts.contains(&inst) {
                if print_removed {
                    log::info!(
                        "[DCE] Removing dead instruction: {:?} = {:?}",
                        inst,
                        func.dfg.instructions[inst]
                    );
                }
                // 原子地完成断开引用和 Nop 替换
                func.dfg.remove_inst(inst);
                changed = true;
            }
        }
    }

    if changed {
        // 4. Compact blocks: 从 Layout 的指令列表中移除 Nops
        for block in &func.layout.block_order {
            func.layout.blocks[*block]
                .insts
                .retain(|&inst| !matches!(func.dfg.instructions[inst], InstructionData::Nop));
        }
    }

    changed
}
