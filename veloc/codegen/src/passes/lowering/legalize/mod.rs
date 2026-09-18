pub mod info;

pub use info::*;

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use crate::target::arch::TargetLegalizer;
use veloc_lir::MachineFunction;

pub struct Legalizer<'a> {
    target: &'a dyn TargetLegalizer,
}

impl<'a> Legalizer<'a> {
    pub fn new(target: &'a dyn TargetLegalizer) -> Self {
        Self { target }
    }

    pub fn legalize(&self, mfunc: &mut MachineFunction) -> Result<bool> {
        // Process expansions in program order, including generic instructions
        // produced by other rules. A single forward scan is not a legalizer.
        // One budget for the entire run: a rule creating blocks must not reset
        // the convergence guard by moving its next rewrite to another block.
        let budget = mfunc.inst_count().max(1).saturating_mul(1024);
        let mut rewrites = 0;
        let mut trace = alloc::collections::VecDeque::new();
        let mut pending = alloc::vec::Vec::new();
        let mut dirty = alloc::collections::BTreeSet::new();
        let mut queued: alloc::collections::VecDeque<_> = mfunc.blocks().collect();
        let mut known: hashbrown::HashSet<_> = mfunc.blocks().collect();
        let mut visited = hashbrown::HashSet::new();
        while !queued.is_empty() || !dirty.is_empty() {
            let block = if let Some(block) = dirty.pop_first() {
                block
            } else {
                queued.pop_front().unwrap()
            };
            if !mfunc.layout().contains_block(block) {
                continue;
            }
            visited.insert(block);
            pending.clear();
            pending.extend(mfunc.block_insts(block).rev());
            while let Some(id) = pending.pop() {
                if mfunc.inst_block(id) != Some(block) {
                    continue;
                }
                let inst = &mfunc.inst(id);
                if inst.is_invalid() {
                    continue;
                }
                let action = if inst.is_generic() {
                    let query = Query::from_inst(inst, mfunc)?;
                    self.target.legalize_action(&query)?
                } else {
                    self.target.legalize_target(inst)?
                };
                match action {
                    None => {
                        return Err(Error::codegen(alloc::format!(
                            "missing legalization rule for {:?}",
                            inst.opcode()
                        )));
                    }
                    Some(LegalizeAction::Legal) => {}
                    Some(action) => {
                        let rule = action.name();
                        if rewrites == budget {
                            return Err(Error::codegen(alloc::format!(
                                "legalization did not converge after {budget} rewrites; recent rules: {trace:?}; next: {:?} {rule:?}",
                                inst.opcode(),
                            )));
                        }
                        rewrites += 1;
                        if trace.len() == 16 {
                            trace.pop_front();
                        }
                        trace.push_back((id, inst.opcode(), rule));
                        let (result, changes) = mfunc.track_edits(|f| action.apply(id, f));
                        let LegalizeResult::Replace(output) = result?;
                        for &owner in &changes.blocks {
                            if !mfunc.layout().contains_block(owner) {
                                continue;
                            }
                            if known.insert(owner) {
                                queued.push_back(owner);
                            }
                            if visited.contains(&owner) {
                                dirty.insert(owner);
                            }
                        }
                        for changed in changes.insts {
                            if let Some(owner) = mfunc.inst_block(changed) {
                                if visited.contains(&owner) {
                                    dirty.insert(owner);
                                }
                            }
                        }
                        mfunc.editor().replace_with(id, &output);
                        pending.extend(output.into_iter().rev());
                    }
                }
            }
            if known.len() != mfunc.num_blocks() {
                for block in mfunc.blocks() {
                    if known.insert(block) {
                        queued.push_back(block);
                    }
                }
            }
        }
        Ok(rewrites != 0)
    }
}
