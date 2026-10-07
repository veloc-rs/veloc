//! Specialize small dispatch paths on constant incoming block arguments.
//! Only closed, constant-fed parameter networks of branch tables seed the
//! transformation. Ordinary induction variables do not trigger loop unrolling.
mod ssa;

use crate::{FunctionPass, OptConfig, PassOutcome, Profile, evaluate};
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{
    Block, FuncBody, Inst, InstView, SuccessorData, Value, ValueDef, function::EdgeRef,
};

pub struct ThreadingPass;

const MAX_BLOCK_INSTS: usize = 12;
const MAX_CLONED_INSTS: usize = 512;

fn state_parameters(f: &FuncBody) -> HashSet<Value> {
    let mut inputs = HashMap::<Value, Vec<Value>>::new();
    let mut selectors = Vec::new();
    for block in f.layout().block_order() {
        let inst = f.layout().last_inst(block).expect("terminated block");
        let view = f.dfg().inst(inst);
        if let InstView::BrTable { index, .. } = view {
            selectors.push(index);
        }
        view.visit_successors(|edge| {
            for (&param, &arg) in f.dfg().block_params(edge.block).iter().zip(edge.args) {
                inputs.entry(param).or_default().push(arg);
            }
        });
    }
    let mut states = HashSet::new();
    for selector in selectors {
        let mut seen = HashSet::new();
        let mut pending = vec![selector];
        let mut constants = HashSet::new();
        let mut closed = true;
        while let Some(value) = pending.pop() {
            if f.dfg().as_scalar_const(value).is_some() {
                constants.insert(value);
                continue;
            }
            if !seen.insert(value) {
                continue;
            }
            if let Some(args) = inputs.get(&value) {
                pending.extend(args);
            } else {
                closed = false;
                break;
            }
        }
        if closed && constants.len() >= 2 {
            states.extend(seen);
        }
    }
    states
}

#[derive(PartialEq, Eq, Hash)]
struct Version {
    block: Block,
    constants: Vec<(Value, Value)>,
}

struct Work {
    queue: VecDeque<Block>,
    queued: HashSet<Block>,
}
impl Work {
    fn push(&mut self, block: Block) {
        if self.queued.insert(block) {
            self.queue.push_back(block);
        }
    }
    fn pop(&mut self) -> Option<Block> {
        let block = self.queue.pop_front()?;
        self.queued.remove(&block);
        Some(block)
    }
}

impl FunctionPass for ThreadingPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "ThreadingPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, profile: &Profile) -> PassOutcome {
        let mut states = state_parameters(am.function());
        if states.is_empty() {
            return PassOutcome::Unchanged;
        }
        let f = am.function_mut();
        // Dead blocks need not obey inter-block dominance. Remove them before
        // tracing live definitions backward through predecessor edges.
        let mut changed = f.edit().remove_unreachable();
        let initial = f
            .layout()
            .block_order()
            .map(|b| f.layout().block_insts(b).count())
            .sum::<usize>();
        // Pay for both code growth and compilation. A budget stop preserves
        // the unmodified dispatch path for all unhandled input combinations.
        let budget = initial.min(MAX_CLONED_INSTS);
        let mut remaining = budget;
        let transported = ssa::localize(f);
        changed |= !transported.is_empty();
        for (old, param) in transported {
            if states.contains(&old) {
                states.insert(param);
            }
        }
        let original: HashSet<_> = f.layout().block_order().collect();
        let mut work = Work {
            queue: VecDeque::new(),
            queued: HashSet::new(),
        };
        for block in f.layout().block_order() {
            work.push(block);
        }
        let mut versions = HashMap::<Version, Block>::new();
        let mut clones = 0;
        let mut folded = 0;
        while let Some(block) = work.pop() {
            if !original.contains(&block) {
                continue;
            }
            let params = f.dfg().block_params(block).to_vec();
            if !params.iter().any(|p| states.contains(p)) {
                continue;
            }
            let insts: Vec<_> = f.layout().block_insts(block).collect();
            if insts.len() > MAX_BLOCK_INSTS
                || insts.iter().any(|&i| {
                    let op = f.dfg().inst(i).opcode();
                    op.transfers_ownership() || matches!(f.dfg().inst(i), InstView::Alloca { .. })
                })
            {
                continue;
            }
            let incoming = incoming(f, block);
            for (edge, args) in incoming {
                let constants: Vec<_> = params
                    .iter()
                    .copied()
                    .zip(args.iter().copied())
                    .filter(|(p, v)| states.contains(p) && f.dfg().as_scalar_const(*v).is_some())
                    .collect();
                if constants.is_empty() {
                    continue;
                }
                let key = Version { block, constants };
                let clone = if let Some(&clone) = versions.get(&key) {
                    clone
                } else {
                    let cost = insts.len().max(1);
                    if cost > remaining {
                        continue;
                    }
                    remaining -= cost;
                    let (clone, branches) = duplicate(f, &key, &params, &insts);
                    versions.insert(key, clone);
                    clones += 1;
                    folded += branches;
                    for &succ in f.cfg().succs(clone) {
                        work.push(succ);
                    }
                    clone
                };
                f.edit().redirect_edge(edge, clone, &args);
                changed = true;
            }
        }
        f.edit().remove_unreachable();
        profile.count("threading.cloned_blocks", clones);
        profile.count("threading.folded_branches", folded);
        profile.count("threading.growth_cost", (budget - remaining) as u64);
        if changed {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}

fn incoming(f: &FuncBody, block: Block) -> Vec<(EdgeRef, Vec<Value>)> {
    let mut incoming = Vec::new();
    for &pred in f.cfg().preds(block) {
        let inst = f.layout().last_inst(pred).expect("terminated predecessor");
        let mut index = 0;
        f.dfg().inst(inst).visit_successors(|edge| {
            if edge.block == block {
                incoming.push((EdgeRef { inst, index }, edge.args.to_vec()));
            }
            index += 1;
        });
    }
    incoming
}

fn duplicate(
    f: &mut FuncBody,
    version: &Version,
    params: &[Value],
    insts: &[Inst],
) -> (Block, u64) {
    let block = f.edit().create_block();
    f.edit().append_block(block);
    let constants: HashMap<_, _> = version.constants.iter().copied().collect();
    let mut values = HashMap::new();
    for &param in params {
        let ty = f.dfg().value_type(param);
        let new = f.edit().append_block_param(block, ty);
        values.insert(param, constants.get(&param).copied().unwrap_or(new));
    }
    let mut branches = 0;
    for &inst in insts {
        let args: Vec<_> = f
            .dfg()
            .operands(inst)
            .iter()
            .map(|v| {
                values.get(v).copied().unwrap_or_else(|| {
                    debug_assert!(matches!(
                        f.dfg().value_def(*v),
                        ValueDef::Const(_) | ValueDef::FunctionParam(_)
                    ));
                    *v
                })
            })
            .collect();
        let results = f.dfg().inst_results(inst).to_vec();
        let types: Vec<_> = results.iter().map(|&v| f.dfg().value_type(v)).collect();
        if let Some(folds) = evaluate::reduce(f.dfg().inst_fields(inst), &args, &types, |v| {
            f.dfg().as_scalar_const(v)
        }) {
            for (&old, fold) in results.iter().zip(folds) {
                let new = match fold {
                    evaluate::Fold::Operand(i) => args[i],
                    evaluate::Fold::Constant(c) => f.edit().constant(c.into()),
                };
                values.insert(old, new);
            }
            continue;
        }
        let chosen = match f.dfg().inst(inst) {
            InstView::Br { condition, .. } => f
                .dfg()
                .as_scalar_const(values.get(&condition).copied().unwrap_or(condition))
                .map(|c| usize::from(c.to_bits() == 0)),
            InstView::BrTable { index, table } => f
                .dfg()
                .as_scalar_const(values.get(&index).copied().unwrap_or(index))
                .map(|c| (c.to_bits() as usize).min(table.len() - 1)),
            _ => None,
        };
        if let Some(chosen) = chosen {
            let mut dest = None;
            let mut index = 0;
            f.dfg().inst(inst).visit_successors(|edge| {
                if index == chosen {
                    let args: Vec<_> = edge
                        .args
                        .iter()
                        .map(|v| values.get(v).copied().unwrap_or(*v))
                        .collect();
                    dest = Some(SuccessorData::new(edge.block, &args));
                }
                index += 1;
            });
            let dest = dest.expect("selected successor");
            f.edit().append_inst(block, |w| w.jump(dest.as_view()), &[]);
            branches += 1;
        } else {
            let new = f
                .edit()
                .append_inst(block, |w| w.copy_with_operands(inst, &args), &types);
            for (&old, &new) in results.iter().zip(f.dfg().inst_results(new)) {
                values.insert(old, new);
            }
        }
    }
    (block, branches)
}
