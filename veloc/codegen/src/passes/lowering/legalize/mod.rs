pub mod info;

pub use info::*;

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use crate::target::TargetLegalizer;
use veloc_lir::MachineFunction;

pub struct Legalizer<'a> {
    target: &'a dyn TargetLegalizer,
}

impl<'a> Legalizer<'a> {
    pub fn new(target: &'a dyn TargetLegalizer) -> Self {
        Self { target }
    }

    pub fn legalize(&self, mfunc: &mut MachineFunction) -> Result<bool> {
        use alloc::collections::VecDeque;
        const REWRITES_PER_INST: usize = 1024;
        const TRACE_LENGTH: usize = 16;

        // Queries depend only on instruction data and immutable value types.
        // CFG changes alone cannot change legality; placement events report the
        // affected instructions, including instructions entering new blocks.
        let budget = mfunc.inst_count().max(1).saturating_mul(REWRITES_PER_INST);
        let mut pending: VecDeque<_> = mfunc
            .blocks()
            .flat_map(|block| mfunc.block_insts(block))
            .collect();
        let mut queued: hashbrown::HashSet<_> = pending.iter().copied().collect();
        let mut rewrites = 0;
        let mut trace = VecDeque::new();
        while let Some(id) = pending.pop_front() {
            queued.remove(&id);
            if mfunc.inst_block(id).is_none() {
                continue;
            }
            let inst = mfunc.inst(id);
            // Target nodes belong to selection/expansion and final emission,
            // not generic instruction legalization.
            if !inst.is_generic() || inst.is_invalid() {
                continue;
            }
            let opcode = inst.opcode();
            let query = Query::from_inst(inst, mfunc.vregs())?;
            let action = self.target.legalize_action(&query)?.ok_or_else(|| {
                Error::codegen(alloc::format!("missing legalization rule for {opcode:?}"))
            })?;
            let LegalizeAction::Rewrite(rewrite) = action else {
                continue;
            };
            let rule = rewrite.name;
            if rewrites == budget {
                return Err(Error::codegen(alloc::format!(
                    "legalization did not converge after {budget} rewrites; recent rules: {trace:?}; next: {id:?} {opcode:?} {rule:?}"
                )));
            }
            let (result, changes) = mfunc.editor().track(|f| rewrite.apply(id, f));
            result?;
            if changes.insts.is_empty() && changes.blocks.is_empty() {
                return Err(Error::codegen(alloc::format!(
                    "legalization rule {rule:?} made no edits for {id:?} {opcode:?}"
                )));
            }
            rewrites += 1;
            if trace.len() == TRACE_LENGTH {
                trace.pop_front();
            }
            trace.push_back((id, opcode, rule));
            // A successful callback is not proof that its surviving root is
            // legal. Re-query it even when only another instruction was edited.
            for changed in changes.insts.into_iter().chain(core::iter::once(id)) {
                if mfunc.inst_block(changed).is_some() && queued.insert(changed) {
                    pending.push_back(changed);
                }
            }
        }
        Ok(rewrites != 0)
    }
}
