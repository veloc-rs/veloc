//! Propagate proven result bits across direct calls within a bound module.
//! Like inlining, this assumes a definition binds calls using its FuncId.
//! Imports and indirect calls remain unknown. Recursive cycles start unknown
//! and gain only facts proved from their bodies, without optimistic assumptions.
use super::{KnownBits, fact, known_bits_with_returns, simplify_conversions};
use crate::{ModulePass, OptConfig, PassOutcome, Profile};
use cranelift_entity::SecondaryMap;
use std::collections::VecDeque;
use veloc_mir::{FuncId, InstView, Module};

pub struct ReturnBitsPass;

impl ModulePass for ReturnBitsPass {
    fn name(&self) -> &'static str {
        "ReturnBitsPass"
    }

    fn run(&self, module: &mut Module, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let mut summaries = SecondaryMap::<FuncId, Vec<KnownBits>>::new();
        let mut callers = SecondaryMap::<FuncId, Vec<FuncId>>::new();
        let mut pending = VecDeque::new();
        let mut queued = SecondaryMap::<FuncId, bool>::new();
        for (id, function) in module.functions() {
            let Some(body) = function.body else { continue };
            summaries[id] = vec![
                KnownBits::default();
                module.signatures()[function.decl.signature].returns().len()
            ];
            pending.push_back(id);
            queued[id] = true;
            for block in body.layout().block_order() {
                for inst in body.layout().block_insts(block) {
                    if let InstView::Call { func_id, .. } = body.dfg().inst(inst) {
                        callers[func_id].push(id);
                    }
                }
            }
        }
        while let Some(id) = pending.pop_front() {
            queued[id] = false;
            let body = module.function(id).body.unwrap();
            let insts: Vec<_> = body
                .layout()
                .block_order()
                .flat_map(|b| body.layout().block_insts(b))
                .collect();
            let facts = known_bits_with_returns(body, &insts, &summaries);
            let mut summary: Option<Vec<KnownBits>> = None;
            for &inst in &insts {
                if let InstView::Return { values } = body.dfg().inst(inst) {
                    let bits: Vec<_> = values.iter().map(|&v| fact(body, &facts, v)).collect();
                    if let Some(summary) = &mut summary {
                        for (result, bits) in summary.iter_mut().zip(bits) {
                            result.zero &= bits.zero;
                            result.one &= bits.one;
                        }
                    } else {
                        summary = Some(bits);
                    }
                }
            }
            let Some(summary) = summary else { continue };
            if summary == summaries[id] {
                continue;
            }
            summaries[id] = summary;
            for &caller in &callers[id] {
                if !queued[caller] {
                    queued[caller] = true;
                    pending.push_back(caller);
                }
            }
        }
        let mut changed = 0;
        for (_, body) in module.bodies_mut() {
            let insts: Vec<_> = body
                .layout()
                .block_order()
                .flat_map(|b| body.layout().block_insts(b))
                .collect();
            let facts = known_bits_with_returns(body, &insts, &summaries);
            for &inst in insts.iter().rev() {
                changed += u64::from(simplify_conversions(body, inst, &facts));
            }
        }
        metrics.count("return_bits.simplified", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}
