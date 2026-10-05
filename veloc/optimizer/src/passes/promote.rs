//! Promote initialized scalar cells in nonescaping entry allocations to SSA.
//! Aggregate fields are independent cells when their byte ranges do not overlap.
//! Unknown/dynamic addressing, escaping pointers and volatile accesses retain
//! memory. Block parameters are pruned by the ordinary parameter/DCE passes.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{BTreeMap, HashSet};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Inst, InstView, Type, function::EdgeRef};

pub struct PromotePass;
#[derive(Clone)]
struct Cell {
    object: Inst,
    offset: u32,
    bytes: u32,
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
        let Some(layout) = config.data_layout else {
            return PassOutcome::Unchanged;
        };
        let f = am.function_mut();
        let entry = f.entry_block();
        let blocks: Vec<_> = f.layout().block_order().collect();
        let insts: Vec<_> = blocks
            .iter()
            .flat_map(|&b| f.layout().block_insts(b))
            .collect();
        // Follow every use of each allocation. Bounded address recognition must
        // never turn failure into a proof of nonescape.
        let mut escaped = HashSet::new();
        for &allocation in &insts {
            if !matches!(f.dfg().inst(allocation), InstView::Alloca { .. }) {
                continue;
            }
            let mut seen = HashSet::new();
            let mut pending = vec![f.dfg().first_result(allocation).unwrap()];
            while let Some(pointer) = pending.pop() {
                if !seen.insert(pointer) {
                    continue;
                }
                for site in f.dfg().uses(pointer) {
                    let inst = site.inst();
                    let access = inst.memory_access(f.dfg());
                    match f.dfg().inst(inst) {
                        InstView::PtrOffset { .. } => {
                            pending.extend_from_slice(f.dfg().inst_results(inst))
                        }
                        _ if access.is_some_and(|a| {
                            a.ptr == pointer
                                && a.stored != Some(pointer)
                                && !a.flags.is_volatile()
                                && f.stack_access(a, &layout).is_some()
                        }) => {}
                        _ => {
                            escaped.insert(allocation);
                        }
                    }
                }
            }
        }
        let mut cells = BTreeMap::<(Inst, u32), Cell>::new();
        for &inst in &insts {
            let Some(access) = inst.memory_access(f.dfg()) else {
                continue;
            };
            let Some((object, offset)) = f.stack_access(access, &layout) else {
                continue;
            };
            let bytes = access.bytes(&layout).unwrap();
            let cell = cells.entry((object, offset)).or_insert(Cell {
                object,
                offset,
                bytes,
                ty: access.ty,
                accesses: vec![],
            });
            if cell.bytes != bytes || cell.ty != access.ty {
                escaped.insert(object);
            }
            cell.accesses.push(inst);
        }
        let cells: Vec<_> = cells.into_values().collect();
        for pair in cells.windows(2) {
            if pair[0].object == pair[1].object && pair[0].offset + pair[0].bytes > pair[1].offset {
                escaped.insert(pair[0].object);
            }
        }
        let mut changed = 0;
        for cell in cells {
            if escaped.contains(&cell.object) {
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
