//! Trace required computations through instructions and block parameters, then
//! remove the unmarked set together, including dead loop-carried computations.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use cranelift_entity::SecondaryMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::text::printer::InstPrinter;
use veloc_mir::{FuncBody, Inst, Value, ValueDef};

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
        // Candidate sessions must have released their temporary uses before DCE.
        let live = Liveness::compute(func);
        let dead: Vec<_> = func
            .layout()
            .block_order()
            .flat_map(|block| func.layout().block_insts(block))
            .filter(|&inst| !live.instructions[inst])
            .collect();
        if config.is_debug_enabled(DCE) {
            let mut text = String::new();
            let printer = InstPrinter::new(func.dfg(), None);
            for &inst in &dead {
                text.clear();
                if printer.fmt_inst_with_results(&mut text, inst).is_ok() {
                    log::info!("[DCE] Removing: {}", text);
                }
            }
        }

        // Remove dead edge arguments before erasing their definitions. Remaining
        // uses of dead parameters belong to the instruction set erased below.
        let mut removed_params = 0;
        let blocks: Vec<_> = func
            .layout()
            .block_order()
            .filter(|&block| block != func.entry_block())
            .collect();
        for block in blocks {
            let keep: Vec<_> = func
                .dfg()
                .block_params(block)
                .iter()
                .map(|&param| live.values[param])
                .collect();
            let count = keep.iter().filter(|&&keep| !keep).count();
            if count != 0 {
                func.edit().retain_block_params(block, &keep);
                removed_params += count;
            }
        }
        func.edit().erase_insts(&dead);
        metrics.count("dce.removed_params", removed_params as u64);
        metrics.count("dce.removed_insts", dead.len() as u64);
        if dead.is_empty() && removed_params == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}

#[derive(Default)]
struct Liveness {
    instructions: SecondaryMap<Inst, bool>,
    values: SecondaryMap<Value, bool>,
    pending: Vec<Value>,
}

impl Liveness {
    fn compute(func: &FuncBody) -> Self {
        let mut live = Self::default();
        let mut incoming = SecondaryMap::<Value, Vec<Value>>::new();
        for block in func.layout().block_order() {
            if let Some(terminator) = func.layout().last_inst(block) {
                func.dfg().inst(terminator).visit_successors(|edge| {
                    for (&param, &arg) in func.dfg().block_params(edge.block).iter().zip(edge.args)
                    {
                        incoming[param].push(arg);
                    }
                });
            }
            for inst in func.layout().block_insts(block) {
                // Preserve control flow, observable effects, trapping operations
                // and ownership transfers according to the IR's erase contract.
                if !func.dfg().inst(inst).can_erase() {
                    live.mark_inst(func, inst);
                }
            }
        }
        while let Some(value) = live.pending.pop() {
            match func.dfg().value_def(value) {
                ValueDef::Inst(inst) => live.mark_inst(func, inst),
                ValueDef::Param(_) => {
                    for &arg in &incoming[value] {
                        live.mark_value(arg);
                    }
                }
                ValueDef::Const(_) => {}
            }
        }
        live
    }

    fn mark_inst(&mut self, func: &FuncBody, inst: Inst) {
        if !self.instructions[inst] {
            self.instructions[inst] = true;
            // Edge arguments are reached through live destination parameters,
            // not merely because their branch instruction is required.
            func.dfg()
                .inst(inst)
                .visit_inputs(|value| self.mark_value(value));
        }
    }

    fn mark_value(&mut self, value: Value) {
        if !self.values[value] {
            self.values[value] = true;
            self.pending.push(value);
        }
    }
}
