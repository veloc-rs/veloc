//! Reuse a dominating load only when every intervening path preserves memory.
//! This needs no language-specific alias assumptions and never speculates a read.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use hashbrown::{HashMap, HashSet};
use veloc_analyzer::{AnalysisManager, graph::DominatorTree};
use veloc_mir::{Block, FuncBody, Inst, Type, Value};

pub struct LoadCsePass;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Address {
    ptr: Value,
    offset: i64,
    ty: Type,
}

struct BlockWrites {
    // Prefix counts let us query either endpoint without scanning instructions.
    prefix: Vec<usize>,
}

impl BlockWrites {
    fn between(&self, start: usize, end: usize) -> bool {
        self.prefix[start] != self.prefix[end]
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
        _: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let func = am.function_mut();
        let order = func.cfg().compute_post_order(func.entry_block());
        let dom = DominatorTree::compute(func.cfg(), func.entry_block());
        let mut positions = HashMap::new();
        let mut writes = HashMap::new();
        for &block in &order {
            let mut prefix = vec![0];
            for (index, inst) in func.layout().block_insts(block).enumerate() {
                positions.insert(inst, (block, index));
                let view = func.dfg().inst(inst);
                let effect = view.memory_effect();
                let barrier = effect.may_write() || effect.may_free() || view.has_volatile_access();
                prefix.push(prefix.last().unwrap() + usize::from(barrier));
            }
            writes.insert(block, BlockWrites { prefix });
        }
        let mut available: HashMap<Address, Vec<Inst>> = HashMap::new();
        let mut removed = Vec::new();
        for block in order.into_iter().rev() {
            let insts: Vec<_> = func.layout().block_insts(block).collect();
            for inst in insts {
                let Some(access) = inst.memory_access(func.dfg()) else {
                    continue;
                };
                if access.stored.is_some() || func.dfg().inst(inst).has_volatile_access() {
                    continue;
                }
                let address = Address {
                    ptr: access.ptr,
                    offset: access.offset,
                    ty: access.ty,
                };
                let candidates = available.entry(address).or_default();
                let previous = candidates.iter().rev().copied().find(|old| {
                    let (definition, start) = positions[old];
                    dom.dominates(definition, block)
                        && preserved(func, &writes, (definition, start + 1), positions[&inst])
                });
                if let Some(old) = previous {
                    let value = func.dfg().first_result(old).unwrap();
                    let result = func.dfg().first_result(inst).unwrap();
                    func.edit().replace_all_uses(result, value);
                    removed.push(inst);
                } else {
                    candidates.push(inst);
                }
            }
        }
        if removed.is_empty() {
            return PreservedAnalyses::all();
        }
        metrics.count("load_cse.removed", removed.len() as u64);
        func.edit().erase_insts(&removed);
        PreservedAnalyses::none()
    }
}

fn preserved(
    func: &FuncBody,
    writes: &HashMap<Block, BlockWrites>,
    (definition, start): (Block, usize),
    (use_block, end): (Block, usize),
) -> bool {
    if definition == use_block {
        return !writes[&definition].between(start, end);
    }
    if writes[&use_block].between(0, end)
        || writes[&definition].between(start, writes[&definition].end())
    {
        return false;
    }
    let mut work = func.cfg().preds(use_block).to_vec();
    let mut visited = HashSet::new();
    while let Some(block) = work.pop() {
        if block == definition || !visited.insert(block) {
            continue;
        }
        if writes[&block].between(0, writes[&block].end()) {
            return false;
        }
        work.extend(func.cfg().preds(block));
    }
    true
}
