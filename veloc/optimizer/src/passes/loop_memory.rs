//! Carry a stored value around a simple loop instead of loading it again.
//! Stores remain at their original positions: potentially aliasing reads still
//! observe every write. No type-based alias assumption is required.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use veloc_analyzer::{
    AnalysisManager,
    graph::{DominatorTree, LoopInfo},
};
use veloc_mir::{InstView, ValueDef, function::EdgeRef};

pub struct LoopMemoryPass;
impl FunctionPass for LoopMemoryPass {
    fn name(&self) -> &'static str {
        "LoopMemoryPass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        _: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let f = am.function_mut();
        let dom = DominatorTree::compute(f.cfg(), f.entry_block());
        let loops = LoopInfo::compute(f.cfg(), &dom);
        let mut changed = 0;
        for &(latch, header) in loops.backedges() {
            if latch == header
                || f.cfg().preds(latch) != [header]
                || f.cfg().succs(latch) != [header]
            {
                continue;
            }
            let preds = f.cfg().preds(header);
            if preds.len() != 2 {
                continue;
            }
            let entry = *preds.iter().find(|&&b| b != latch).unwrap();
            if dom.dominates(header, entry) || f.cfg().succs(entry) != [header] {
                continue;
            }
            let insts: Vec<_> = f
                .layout()
                .block_insts(header)
                .chain(f.layout().block_insts(latch))
                .collect();
            let stores: Vec<_> = insts
                .iter()
                .copied()
                .filter(|&i| matches!(f.dfg().inst(i), InstView::Store { .. }))
                .collect();
            let Some(&store) = stores.first() else {
                continue;
            };
            let InstView::Store {
                ptr,
                offset,
                value,
                flags,
            } = f.dfg().inst(store)
            else {
                unreachable!()
            };
            if flags.is_volatile() {
                continue;
            }
            let ty = f.dfg().value_type(value);
            let owner = match f.dfg().value_def(ptr) {
                ValueDef::Inst(i) => f.layout().inst_block(i),
                ValueDef::Param(b) => Some(b),
                ValueDef::Const(_) => None,
            };
            if owner.is_some_and(|b| !dom.dominates(b, entry)) {
                continue;
            }
            let mut safe = true;
            let mut loads = 0;
            for &inst in &insts {
                let view = f.dfg().inst(inst);
                if view.has_volatile_access() {
                    safe = false;
                    break;
                }
                match view {
                    InstView::Store {
                        ptr: p,
                        offset: o,
                        value: v,
                        ..
                    } => {
                        if p != ptr || o != offset || f.dfg().value_type(v) != ty {
                            safe = false;
                            break;
                        }
                    }
                    InstView::Load {
                        ptr: p, offset: o, ..
                    } if p == ptr
                        && o == offset
                        && f.dfg().value_type(f.dfg().first_result(inst).unwrap()) == ty =>
                    {
                        loads += 1
                    }
                    _ if view.memory_effect().may_write() || view.memory_effect().may_free() => {
                        safe = false;
                        break;
                    }
                    _ => {}
                }
            }
            if !safe || loads == 0 {
                continue;
            }
            // Find the last write before entry; unknown effects block the proof.
            let mut initial = None;
            for inst in f.layout().block_insts(entry).rev() {
                let view = f.dfg().inst(inst);
                if let InstView::Store {
                    ptr: p,
                    offset: o,
                    value: v,
                    flags,
                } = view
                    && p == ptr
                    && o == offset
                    && f.dfg().value_type(v) == ty
                    && !flags.is_volatile()
                {
                    initial = Some(v);
                    break;
                }
                if view.has_volatile_access()
                    || view.memory_effect().may_write()
                    || view.memory_effect().may_free()
                {
                    break;
                }
            }
            let Some(initial) = initial else {
                continue;
            };
            let param = f.edit().append_block_param(header, ty);
            let mut current = param;
            for inst in insts {
                match f.dfg().inst(inst) {
                    InstView::Store { value, .. } => current = value,
                    InstView::Load {
                        ptr: p, offset: o, ..
                    } if p == ptr
                        && o == offset
                        && f.dfg().value_type(f.dfg().first_result(inst).unwrap()) == ty =>
                    {
                        f.edit().replace_results(inst, &[current]);
                        changed += 1;
                    }
                    _ => {}
                }
            }
            for (block, value) in [(entry, initial), (latch, current)] {
                let inst = f.layout().last_inst(block).unwrap();
                let mut edges = Vec::new();
                let mut index = 0;
                f.dfg().inst(inst).visit_successors(|s| {
                    if s.block == header {
                        let mut args = s.args.to_vec();
                        args.push(value);
                        edges.push((index, args));
                    }
                    index += 1;
                });
                for (index, args) in edges {
                    f.edit()
                        .redirect_edge(EdgeRef { inst, index }, header, &args);
                }
            }
        }
        metrics.count("loop_memory.forwarded", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}
