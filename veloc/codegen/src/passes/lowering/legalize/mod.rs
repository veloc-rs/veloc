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
        let mut owners = alloc::vec![None; mfunc.inst_count()];
        for (block, data) in mfunc.blocks.iter().enumerate() {
            for id in &data.insts {
                owners[id.as_u32() as usize] = Some(block);
            }
        }
        let mut dirty = alloc::collections::BTreeSet::new();
        let mut fresh = 0;
        while fresh < mfunc.blocks.len() || !dirty.is_empty() {
            let block = if let Some(block) = dirty.pop_first() {
                block
            } else {
                let block = fresh;
                fresh += 1;
                block
            };
            mfunc.rewrite_block(block, |cursor| {
                pending.clear();
                pending.push(cursor.current_inst_id());
                cursor.detach_current();
                while let Some(id) = pending.pop() {
                    let inst = &cursor.mfunc().inst(id);
                    if inst.is_invalid() {
                        continue;
                    }
                    if inst.generic_opcode().is_none() {
                        cursor.emit(id);
                        continue;
                    }
                    let query = Query::from_inst(inst, cursor.mfunc())?;
                    match self.target.legalize_action(&query)? {
                        None => {
                            let opcode = query.opcode;
                            let operands = (&query.results, &query.inputs);
                            return Err(Error::codegen(alloc::format!(
                                "missing legalization rule for {opcode:?} with signature {operands:?}"
                            )));
                        }
                        Some(LegalizeAction::Legal) => cursor.emit(id),
                        Some(action) => {
                            let rule = action.name();
                            if rewrites == budget {
                                return Err(Error::codegen(alloc::format!(
                                    "legalization did not converge after {budget} rewrites; recent rules: {trace:?}; next: {:?} {rule:?}",
                                    inst.opcode(),
                                )));
                            }
                            rewrites += 1;
                            if trace.len() == 16 { trace.pop_front(); }
                            trace.push_back((id, inst.opcode(), rule));
                            let (result, changes) = cursor.mfunc_mut().track_inst_changes(|f| action.apply(id, f));
                            let LegalizeResult::Replace(output) = result?;
                            for changed in changes {
                                if let Some(Some(owner)) = owners.get(changed.as_u32() as usize) {
                                    if *owner < fresh { dirty.insert(*owner); }
                                }
                            }
                            // Rules may rewrite the same ID in place. Preserve it
                            // in that case and check its new form on the worklist.
                            if !output.contains(&id) {
                                cursor.mfunc_mut().invalidate_inst(id);
                            }
                            pending.extend(output.into_iter().rev());
                        }
                    }
                }
                Ok(())
            })?;
            owners.resize(mfunc.inst_count(), None);
            for id in &mfunc.blocks[block].insts {
                owners[id.as_u32() as usize] = Some(block);
            }
        }
        Ok(rewrites != 0)
    }
}
