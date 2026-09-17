//! Generated local rewrites: constant evaluation and reviewed algebraic rules.
//!
//! Fold constant expressions and simplify identities to a local fixed point.
//! For example: `iconst 1 + iconst 2` -> `iconst 3`, and `x + 0` -> `x`.

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use alloc::collections::VecDeque;
use cranelift_entity::SecondaryMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::Function;

const SIMPLIFY: &str = "simplify";

pub struct SimplifyPass;

impl FunctionPass for SimplifyPass {
    fn name(&self) -> &str {
        "SimplifyPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        let changed = run_simplify(
            am.function_mut(),
            config.is_debug_enabled(SIMPLIFY),
            metrics,
        );
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_simplify(func: &mut Function, debug: bool, metrics: &mut Metrics) -> bool {
    let mut queue = VecDeque::new();
    let mut queued = SecondaryMap::<veloc_mir::Inst, bool>::new();
    for block in func.layout().block_order() {
        for inst in func.layout().block_insts(block) {
            queue.push_back(inst);
            queued[inst] = true;
        }
    }
    let mut rewritten = 0u64;
    while let Some(inst) = queue.pop_front() {
        queued[inst] = false;
        if func.layout().inst_block(inst).is_none() {
            continue;
        }
        if let Some(affected) = crate::rewrite::rewrite(func, inst) {
            if debug {
                log::info!("Rewrote instruction {}", inst);
            }
            rewritten += 1;
            for user in affected {
                if !queued[user] && func.layout().inst_block(user).is_some() {
                    queued[user] = true;
                    queue.push_back(user);
                }
            }
        }
    }
    if rewritten != 0 {
        metrics.add("simplify.rewritten_insts", rewritten);
    }
    rewritten != 0
}
