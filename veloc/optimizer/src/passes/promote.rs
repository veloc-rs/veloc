//! Promote initialized scalar cells in nonescaping stack slots to SSA.
//! Aggregate fields are independent cells when their byte ranges do not overlap.
//! Unknown/dynamic addressing, escaping pointers and volatile accesses retain
//! memory. Block parameters are pruned by the ordinary parameter/DCE passes.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{BTreeMap, HashSet};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Inst, InstView, Type, Value, function::EdgeRef, memory::ByteRange};

pub struct PromotePass;
#[derive(Clone)]
struct Cell {
    range: ByteRange,
    ty: Type,
    accesses: Vec<Inst>,
}
impl FunctionPass for PromotePass {
    fn name(&self) -> &'static str {
        "PromotePass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        let layout = config.data_layout;
        let f = am.function_mut();
        let entry = f.entry_block();
        let blocks: Vec<_> = f.layout().block_order().collect();
        let insts: Vec<_> = blocks
            .iter()
            .flat_map(|&b| f.layout().block_insts(b))
            .collect();
        // Follow derived pointers so every use of each slot is accounted for.
        let mut ineligible = HashSet::new();
        for &allocation in f.stack_slots() {
            let mut seen = HashSet::new();
            let mut pending = vec![allocation];
            while let Some(pointer) = pending.pop() {
                if !seen.insert(pointer) {
                    continue;
                }
                for site in f.dfg().uses(pointer) {
                    let inst = site.inst();
                    let flags = match f.dfg().inst(inst) {
                        InstView::PtrOffset { .. } => {
                            pending.extend_from_slice(f.dfg().inst_results(inst));
                            continue;
                        }
                        InstView::Load { ptr, flags, .. } if ptr == pointer => flags,
                        InstView::Store {
                            ptr, value, flags, ..
                        } if ptr == pointer && value != pointer => flags,
                        _ => {
                            ineligible.insert(allocation);
                            continue;
                        }
                    };
                    if flags.is_volatile()
                        || f.access(inst)
                            .and_then(|a| a.range(&layout))
                            .is_none_or(|r| r.base != allocation)
                    {
                        ineligible.insert(allocation);
                    }
                }
            }
        }
        let mut cells = BTreeMap::<(Value, i64), Cell>::new();
        for &inst in &insts {
            let Some(access) = f.access(inst) else {
                continue;
            };
            let Some(range) = access.range(&layout) else {
                continue;
            };
            if f.dfg().stack_slot_layout(range.base).is_none() {
                continue;
            }
            let ty = access.ty;
            let cell = cells.entry((range.base, range.offset)).or_insert(Cell {
                range,
                ty,
                accesses: vec![],
            });
            if cell.range.bytes != range.bytes || cell.ty != ty {
                ineligible.insert(range.base);
            }
            cell.accesses.push(inst);
        }
        let cells: Vec<_> = cells.into_values().collect();
        for pair in cells.windows(2) {
            if pair[0].range.overlaps(&pair[1].range) {
                ineligible.insert(pair[0].range.base);
            }
        }
        let mut changed = 0;
        for cell in cells {
            if ineligible.contains(&cell.range.base) {
                continue;
            }
            let accesses: HashSet<_> = cell.accesses.iter().copied().collect();
            let mut initialized = false;
            let mut valid = true;
            for inst in f
                .layout()
                .block_insts(entry)
                .filter(|i| accesses.contains(i))
            {
                match f.dfg().inst(inst) {
                    InstView::Store { .. } => initialized = true,
                    _ if !initialized => valid = false,
                    _ => {}
                }
            }
            if !initialized || !valid {
                continue;
            }
            let mut params = BTreeMap::new();
            for &block in &blocks {
                if block != entry {
                    let param = f.edit().append_block_param(block, cell.ty);
                    params.insert(block, param);
                }
            }
            let mut ends = BTreeMap::new();
            let mut dead = Vec::new();
            for &block in &blocks {
                let mut current = params.get(&block).copied();
                let insts: Vec<_> = f
                    .layout()
                    .block_insts(block)
                    .filter(|i| accesses.contains(i))
                    .collect();
                for inst in insts {
                    match f.dfg().inst(inst) {
                        InstView::Store { value, .. } => current = Some(value),
                        InstView::Load { .. } => {
                            let value = f.dfg().first_result(inst).unwrap();
                            f.edit().replace_all_uses(value, current.unwrap());
                        }
                        _ => unreachable!(),
                    }
                    dead.push(inst);
                }
                ends.insert(block, current.unwrap());
            }
            // A store can refer to a load from an earlier/later layout block.
            // Read back stores after all replacements before constructing edges.
            for &block in &blocks {
                let last_store = f
                    .layout()
                    .block_insts(block)
                    .filter(|i| accesses.contains(i))
                    .filter_map(|i| {
                        if let InstView::Store { value, .. } = f.dfg().inst(i) {
                            Some(value)
                        } else {
                            None
                        }
                    })
                    .last();
                if let Some(value) = last_store {
                    ends.insert(block, value);
                }
                let term = f.layout().last_inst(block).unwrap();
                let mut edges = Vec::new();
                let mut index = 0;
                f.dfg().inst(term).visit_successors(|e| {
                    let mut args = e.args.to_vec();
                    args.push(ends[&block]);
                    edges.push((index, e.block, args));
                    index += 1;
                });
                for (index, target, args) in edges {
                    f.edit()
                        .redirect_edge(EdgeRef { inst: term, index }, target, &args);
                }
            }
            f.edit().erase_insts(&dead);
            changed += 1;
        }
        metrics.count("promote.cells", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}
