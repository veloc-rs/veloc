use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::FuncBody;
use veloc_mir::inst::Inst;
use veloc_mir::text::printer::InstPrinter;

const DCE: &str = "dce";

pub struct DcePass;

impl FunctionPass for DcePass {
    fn name(&self) -> &'static str {
        "DcePass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let func = am.function_mut();
        let print_removed = config.is_debug_enabled(DCE);
        // Candidate sessions must have released their temporary uses before DCE.
        // Seed once; subsequent checks only visit definitions losing a use.
        let mut work: Vec<Inst> = func
            .layout()
            .block_order()
            .flat_map(|block| func.layout().block_insts(block))
            .filter(|&inst| is_dead(func, inst))
            .collect();
        let mut removed = 0;
        let mut text = String::new();
        while let Some(inst) = work.pop() {
            // Repeated operands or users may enqueue the same definition. Layout
            // membership guards erased IDs without a second membership table.
            if func.layout().inst_block(inst).is_none() || !is_dead(func, inst) {
                continue;
            }
            if print_removed {
                text.clear();
                let printer = InstPrinter::new(func.dfg(), None);
                if printer.fmt_inst_with_results(&mut text, inst).is_ok() {
                    log::info!("[DCE] Removing: {}", text);
                }
            }
            // Save definitions before erasure clears operands and updates use-def.
            // They are examined only after this instruction's uses are removed.
            work.extend(
                func.dfg()
                    .operands(inst)
                    .iter()
                    .filter_map(|&value| func.dfg().value_inst(value)),
            );
            func.edit().erase_inst(inst);
            removed += 1;
        }
        if removed != 0 {
            metrics.count("dce.removed_insts", removed);
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

fn is_dead(func: &FuncBody, inst: Inst) -> bool {
    func.dfg().inst(inst).can_erase()
        && func
            .dfg()
            .inst_results(inst)
            .iter()
            .all(|&value| func.dfg().uses(value).next().is_none())
}
