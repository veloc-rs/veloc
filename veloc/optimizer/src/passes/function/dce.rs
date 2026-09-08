use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::Function;
use veloc_mir::inst::Inst;
use veloc_mir::text::printer::InstPrinter;
use veloc_mir::types::ValueDef;

const DCE: &str = "dce";

pub struct DcePass;

impl FunctionPass for DcePass {
    fn name(&self) -> &str {
        "DcePass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        let changed = run_dce(am.function_mut(), config.is_debug_enabled(DCE), metrics);
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_dce(func: &mut Function, print_removed: bool, metrics: &mut Metrics) -> bool {
    let mut live_insts = HashSet::new();
    let mut worklist: Vec<Inst> = Vec::new();

    // 1. Identify roots: instructions with side effects
    for block in func.layout().block_order() {
        for &inst in &func.layout().blocks()[*block].insts {
            if func.dfg().inst(inst).has_side_effects() && live_insts.insert(inst) {
                worklist.push(inst);
            }
        }
    }

    // 2. Propagate liveness back through use-def chains
    while let Some(inst) = worklist.pop() {
        for &val in func.dfg().operands(inst) {
            if let ValueDef::Inst(def_inst) = func.dfg().values()[val].def
                && live_insts.insert(def_inst)
            {
                worklist.push(def_inst);
            }
        }
    }

    // 3. Collect dead instructions first, then process them
    let dead_insts: Vec<Inst> = func
        .layout()
        .block_order()
        .iter()
        .flat_map(|block| &func.layout().blocks()[*block].insts)
        .filter(|&&inst| !live_insts.contains(&inst))
        .copied()
        .collect();

    for &inst in &dead_insts {
        if print_removed {
            let printer = InstPrinter::new(func.dfg(), None);
            let mut buf = String::new();
            if let Ok(()) = printer.fmt_inst_with_results(&mut buf, inst) {
                log::info!("[DCE] Removing: {}", buf);
            }
        }
    }

    if !dead_insts.is_empty() {
        func.edit().erase_insts(&dead_insts);
        metrics.add("dce.removed_insts", dead_insts.len() as u64);
    }

    !dead_insts.is_empty()
}
