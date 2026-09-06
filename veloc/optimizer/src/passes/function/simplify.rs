//! Generated local rewrites: constant evaluation and reviewed algebraic rules.
//!
//! Fold constant expressions and simplify identities to a local fixed point.
//! For example: `iconst 1 + iconst 2` -> `iconst 3`, and `x + 0` -> `x`.

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::Function;
use veloc_mir::inst::InstructionData;

const SIMPLIFY: &str = "simplify";

pub struct SimplifyPass;

impl FunctionPass for SimplifyPass {
    fn name(&self) -> &str {
        "SimplifyPass"
    }

    fn run(
        &self,
        func: &mut Function,
        am: &mut AnalysisManager,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        let changed = run_simplify(func, am, config.is_debug_enabled(SIMPLIFY), metrics);
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_simplify(
    func: &mut Function,
    am: &mut AnalysisManager,
    debug: bool,
    metrics: &mut Metrics,
) -> bool {
    let mut changed = false;
    let mut rewritten = 0u64;

    loop {
        let mut pass_changed = false;

        // 获取指令快照进行遍历。
        let insts: Vec<_> = func
            .layout
            .block_order
            .iter()
            .flat_map(|&block| {
                func.layout.blocks[block]
                    .insts
                    .iter()
                    .map(move |&inst| (block, inst))
            })
            .collect();

        for (block, inst) in insts {
            // 检查指令是否有效（之前的 fold 可能已经将其变为 Nop）
            if matches!(func.dfg.instructions[inst], InstructionData::Nop) {
                continue;
            }

            if crate::rewrite::rewrite(func, block, inst, am) {
                if debug {
                    log::info!("Rewrote instruction {}", inst);
                }
                pass_changed = true;
                rewritten += 1;
            }
        }

        if !pass_changed {
            break;
        }
        changed = true;
    }

    if changed {
        metrics.add("simplify.rewritten_insts", rewritten);
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
