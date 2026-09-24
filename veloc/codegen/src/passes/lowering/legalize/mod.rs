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

    /// Explicit checkpoint; this never repairs the input or runs during construction.
    pub fn verify(&self, function: &MachineFunction) -> Result<()> {
        for id in function.blocks().flat_map(|b| function.block_insts(b)) {
            let inst = function.inst(id);
            if function
                .try_call_info(id)
                .is_some_and(|info| info.frame.is_none())
            {
                return Err(Error::codegen(std::format!("unlowered ABI call {id:?}")));
            }
            if inst.is_generic() && !inst.is_call_frame() {
                let query = Query::from_inst(inst, function.vregs())?;
                if !matches!(
                    self.target.legalize_action(&query)?,
                    Some(LegalizeAction::Legal)
                ) {
                    return Err(Error::codegen(std::format!(
                        "illegal instruction at selection boundary: {id:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// New calls enter ABI lowering before their generated transfers are legalized.
    pub fn legalize(
        &self,
        mfunc: &mut MachineFunction,
        mut lower_call: impl FnMut(&mut veloc_lir::FuncEditor<'_>, veloc_lir::InstId) -> Result<()>,
    ) -> Result<bool> {
        use std::collections::VecDeque;
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
            if !inst.is_generic() || inst.is_invalid() || inst.is_call_frame() {
                continue;
            }
            let opcode = inst.opcode();
            if mfunc
                .try_call_info(id)
                .is_some_and(|info| info.frame.is_none())
            {
                if rewrites == budget {
                    return Err(Error::codegen("ABI legalization did not converge"));
                }
                let (result, changes) = mfunc.editor().track(|f| lower_call(f, id));
                result?;
                if mfunc
                    .try_call_info(id)
                    .is_some_and(|info| info.frame.is_none())
                {
                    return Err(Error::codegen("ABI lowering left an unresolved call"));
                }
                rewrites += 1;
                for changed in changes.insts.into_iter().chain(core::iter::once(id)) {
                    if mfunc.inst_block(changed).is_some() && queued.insert(changed) {
                        pending.push_back(changed);
                    }
                }
                continue;
            }
            let query = Query::from_inst(inst, mfunc.vregs())?;
            let action = self.target.legalize_action(&query)?.ok_or_else(|| {
                Error::codegen(std::format!("missing legalization rule for {opcode:?}"))
            })?;
            let LegalizeAction::Rewrite(rewrite) = action else {
                continue;
            };
            // General SSA rewrites cannot change an ABI location or its transfer
            // width. Such changes must be expressed by ABI lowering before this
            // boundary, not by pretending a physical register is a typed value.
            if inst
                .results()
                .iter()
                .chain(inst.inputs())
                .any(|reg| reg.is_preg())
            {
                return Err(Error::codegen(
                    "ABI boundary requires unsupported legalization; lower its value conversion before the boundary",
                ));
            }
            let rule = rewrite.name;
            if rewrites == budget {
                return Err(Error::codegen(std::format!(
                    "legalization did not converge after {budget} rewrites; recent rules: {trace:?}; next: {id:?} {opcode:?} {rule:?}"
                )));
            }
            let (result, changes) = mfunc.editor().track(|f| rewrite.apply(id, f));
            result?;
            if changes.insts.is_empty() && changes.blocks.is_empty() {
                return Err(Error::codegen(std::format!(
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
