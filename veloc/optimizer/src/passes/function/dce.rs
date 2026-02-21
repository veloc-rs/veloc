use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_ir::function::Function;
use veloc_ir::inst::{Inst, InstructionData};
use veloc_ir::printer::InstPrinter;
use veloc_ir::types::ValueDef;

const DCE: &str = "dce";

pub struct DcePass;

impl FunctionPass for DcePass {
    fn name(&self) -> &str {
        "DcePass"
    }

    fn run(
        &self,
        func: &mut Function,
        am: &mut AnalysisManager,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        let changed = run_dce(func, am, config.is_debug_enabled(DCE), metrics);
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_dce(
    func: &mut Function,
    _am: &mut AnalysisManager,
    print_removed: bool,
    metrics: &mut Metrics,
) -> bool {
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

    // 3. Collect dead instructions first, then process them
    let dead_insts: Vec<Inst> = func
        .layout
        .block_order
        .iter()
        .flat_map(|block| &func.layout.blocks[*block].insts)
        .filter(|&&inst| !live_insts.contains(&inst))
        .copied()
        .collect();

    let mut removed_count = 0;
    let mut changed = false;

    for inst in dead_insts {
        if print_removed {
            let printer = InstPrinter::new(&func.dfg, None);
            let mut buf = String::new();
            if let Ok(()) = printer.fmt_inst_with_results(&mut buf, inst) {
                log::info!("[DCE] Removing: {}", buf);
            }
        }
        func.dfg.remove_inst(inst);
        removed_count += 1;
        changed = true;
    }

    if changed {
        metrics.add("dce.removed_insts", removed_count);
        // 4. Compact blocks: 从 Layout 的指令列表中移除 Nops
        for block in &func.layout.block_order {
            func.layout.blocks[*block]
                .insts
                .retain(|&inst| !matches!(func.dfg.instructions[inst], InstructionData::Nop));
        }
    }

    changed
}
