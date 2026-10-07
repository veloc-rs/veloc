//! Reuse a dominating load only when every intervening path preserves memory.
//! This needs no language-specific alias assumptions and never speculates a read.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::{HashMap, HashSet};
use veloc_analyzer::AnalysisManager;
use veloc_mir::memory::{Address, Location};
use veloc_mir::{Block, FuncBody, Inst, InstView, Type};
use veloc_types::DataLayout;

pub struct LoadCsePass;

struct BlockWrites {
    // Prefix counts let us query either endpoint without scanning instructions.
    prefix: Vec<usize>,
    stores: Vec<(usize, Location)>,
}

impl BlockWrites {
    fn between(
        &self,
        start: usize,
        end: usize,
        func: &FuncBody,
        location: Option<Location>,
        layout: Option<&DataLayout>,
    ) -> bool {
        self.prefix[start] != self.prefix[end]
            || self.stores.iter().any(|&(at, store)| {
                (start..end).contains(&at)
                    && match (location, layout) {
                        (Some(load), Some(layout)) => func.may_alias(load, store, layout),
                        _ => true,
                    }
            })
    }
    fn end(&self) -> usize {
        self.prefix.len() - 1
    }
}

impl FunctionPass for LoadCsePass {
    fn name(&self) -> &'static str {
        "LoadCsePass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        let dom = am.take_dominators();
        let func = am.function_mut();
        let order = func.cfg().compute_post_order(func.entry_block());
        let mut positions = HashMap::new();
        let mut writes = HashMap::new();
        for &block in &order {
            let mut prefix = vec![0];
            let mut stores = Vec::new();
            for (index, inst) in func.layout().block_insts(block).enumerate() {
                positions.insert(inst, (block, index));
                let view = func.dfg().inst(inst);
                let effect = view.memory_effect();
                let mut barrier = effect.may_free() || view.has_volatile_access();
                if effect.may_write() {
                    let location = config
                        .data_layout
                        .as_ref()
                        .and_then(|layout| func.memory_location(inst, layout));
                    if !barrier && let Some(location) = location {
                        stores.push((index, location));
                    } else {
                        barrier = true;
                    }
                }
                prefix.push(prefix.last().unwrap() + usize::from(barrier));
            }
            writes.insert(block, BlockWrites { prefix, stores });
        }
        let mut available: HashMap<(Address, Type), Vec<Inst>> = HashMap::new();
        let mut removed = Vec::new();
        for block in order.into_iter().rev() {
            let insts: Vec<_> = func.layout().block_insts(block).collect();
            for inst in insts {
                let InstView::Load { ptr, offset, flags } = func.dfg().inst(inst) else {
                    continue;
                };
                if flags.is_volatile() {
                    continue;
                }
                let Some(address) = func.address(ptr, i64::from(offset)) else {
                    continue;
                };
                let location = config
                    .data_layout
                    .as_ref()
                    .and_then(|layout| func.memory_location(inst, layout));
                let result = func.dfg().first_result(inst).expect("load result");
                let ty = func.dfg().value_type(result);
                let candidates = available.entry((address, ty)).or_default();
                let previous = candidates.iter().rev().copied().find(|old| {
                    let (definition, start) = positions[old];
                    dom.dominates(definition, block)
                        && preserved(
                            func,
                            &writes,
                            (definition, start + 1),
                            positions[&inst],
                            location,
                            config.data_layout.as_ref(),
                        )
                });
                if let Some(old) = previous {
                    let value = func.dfg().first_result(old).unwrap();
                    func.edit().replace_all_uses(result, value);
                    removed.push(inst);
                } else {
                    candidates.push(inst);
                }
            }
        }
        if removed.is_empty() {
            return PassOutcome::Unchanged;
        }
        metrics.count("load_cse.removed", removed.len() as u64);
        func.edit().erase_insts(&removed);
        PassOutcome::Changed
    }
}

fn preserved(
    func: &FuncBody,
    writes: &HashMap<Block, BlockWrites>,
    (definition, start): (Block, usize),
    (use_block, end): (Block, usize),
    location: Option<Location>,
    layout: Option<&DataLayout>,
) -> bool {
    if definition == use_block {
        return !writes[&definition].between(start, end, func, location, layout);
    }
    if writes[&use_block].between(0, end, func, location, layout)
        || writes[&definition].between(start, writes[&definition].end(), func, location, layout)
    {
        return false;
    }
    let mut work = func.cfg().preds(use_block).to_vec();
    let mut visited = HashSet::new();
    while let Some(block) = work.pop() {
        if block == definition || !visited.insert(block) {
            continue;
        }
        if writes[&block].between(0, writes[&block].end(), func, location, layout) {
            return false;
        }
        work.extend(func.cfg().preds(block));
    }
    true
}
