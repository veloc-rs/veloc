//! Structural CFG cleanup: terminator rewrites, block merges and unreachable block removal.
//! No abstract interpretation, path predicates or speculative execution.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{HashSet, VecDeque};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, InstView};

pub struct CfgPass;

#[derive(Default)]
struct Worklist {
    queue: VecDeque<Block>,
    queued: HashSet<Block>,
}
impl Worklist {
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

impl FunctionPass for CfgPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "CfgPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let mut changed = u64::from(f.edit().remove_unreachable());
        let mut work = Worklist::default();
        let mut visits = 0;
        loop {
            for block in f.layout().block_order() {
                work.push(block);
            }
            while let Some(block) = work.pop() {
                if f.layout().contains_block(block) {
                    visits += 1;
                    changed += u64::from(simplify_block(f, block, &mut work));
                }
            }
            // Deleting unreachable predecessors can enable more merges.
            if !f.edit().remove_unreachable() {
                break;
            }
            changed += 1;
        }
        metrics.count("cfg.simplified", changed);
        metrics.count("cfg.block_visits", visits);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

fn simplify_block(f: &mut FuncBody, block: Block, work: &mut Worklist) -> bool {
    let mut edited = false;
    while let Some(tail) = f.layout().last_inst(block) {
        if let Some(rewrite) = crate::rewrite::find_rewrite(f, tail) {
            let old_successors = f.cfg().succs(block).to_vec();
            rewrite.apply(f, tail);
            edited = true;
            // A former successor can now merge into its sole remaining predecessor.
            for successor in old_successors {
                if let &[pred] = f.cfg().preds(successor)
                    && pred != block
                {
                    work.push(pred);
                }
            }
            continue;
        }

        let InstView::Jump { dest } = f.dfg().inst(tail) else {
            break;
        };
        // A merge substitutes the successor's parameters even in other blocks.
        let users = merge_users(f, dest.block);
        if !f.edit().merge_successor(block) {
            break;
        }
        edited = true;
        for user in users {
            work.push(user);
        }
    }

    edited
}

fn merge_users(f: &FuncBody, successor: Block) -> Vec<Block> {
    f.dfg()
        .block_params(successor)
        .iter()
        .flat_map(|&param| {
            f.dfg()
                .uses(param)
                .filter_map(|site| f.layout().inst_block(site.inst()))
        })
        .collect()
}
