pub mod info;
pub mod vm;

pub use info::*;

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use std::collections::VecDeque;
use veloc_lir::function::EditChanges;
use veloc_lir::{FuncEditor, InstId, MachineFunction};

pub struct Legalizer<'a> {
    target: LegalizePolicy<'a>,
}

impl<'a> Legalizer<'a> {
    pub fn new(target: LegalizePolicy<'a>) -> Self {
        Self { target }
    }

    /// Explicit read-only checkpoint. Uses the same matcher as execution.
    pub fn verify(&self, function: &MachineFunction) -> Result<()> {
        for id in function.blocks().flat_map(|b| function.block_insts(b)) {
            if needs_abi(function, id) {
                return Err(Error::codegen(format!("unlowered ABI call {id:?}")));
            }
            let inst = function.inst(id);
            if inst.is_generic() && !inst.is_call_frame() {
                if !matches!(
                    vm::select(self.target, function, id)?,
                    Some((_, vm::Action::Legal))
                ) {
                    return Err(Error::codegen(format!(
                        "illegal instruction at selection boundary: {id:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The worklist schedules changed instructions until all are legal.
    /// ABI lowering remains a separate service.
    pub fn legalize(
        &self,
        function: &mut MachineFunction,
        mut lower_call: impl FnMut(&mut FuncEditor<'_>, InstId) -> Result<()>,
    ) -> Result<bool> {
        let mut modified = false;
        let mut pending: VecDeque<_> = function
            .blocks()
            .flat_map(|block| function.block_insts(block))
            .collect();
        let mut queued: hashbrown::HashSet<_> = pending.iter().copied().collect();
        while let Some(id) = pending.pop_front() {
            queued.remove(&id);
            let Some(changes) = self.step(function, id, &mut lower_call)? else {
                continue;
            };
            modified = true;
            // A surviving root must be checked again even if only its neighbours
            // were edited. Removed instructions need no further processing.
            for changed in changes.insts.into_iter().chain(core::iter::once(id)) {
                if function.inst_block(changed).is_some() && queued.insert(changed) {
                    pending.push_back(changed);
                }
            }
        }
        Ok(modified)
    }
}

fn needs_abi(function: &MachineFunction, id: InstId) -> bool {
    function
        .try_call_info(id)
        .is_some_and(|info| info.frame.is_none())
}

impl Legalizer<'_> {
    fn step(
        &self,
        function: &mut MachineFunction,
        id: InstId,
        lower_call: &mut impl FnMut(&mut FuncEditor<'_>, InstId) -> Result<()>,
    ) -> Result<Option<EditChanges>> {
        if function.inst_block(id).is_none() {
            return Ok(None);
        }
        let inst = function.inst(id);
        if !inst.is_generic() || inst.is_invalid() || inst.is_call_frame() {
            return Ok(None);
        }
        let opcode = inst.opcode();
        // None denotes an unresolved call handed off to ABI lowering, not a
        // missing rule. All other instructions must match a legalization entry.
        let selected = if needs_abi(function, id) {
            None
        } else {
            let selected = vm::select(self.target, function, id)?.ok_or_else(|| {
                Error::codegen(format!("missing legalization rule for {opcode:?}"))
            })?;
            if matches!(selected.1, vm::Action::Legal) {
                return Ok(None);
            }
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
            Some(selected)
        };
        let rule = match selected {
            None => "ABI lowering",
            Some((_, vm::Action::Recipe { name, .. })) => name,
            Some((_, vm::Action::Legal)) => unreachable!("legal instructions do not rewrite"),
        };
        let (result, changes) = function.editor().track(|edit| {
            let Some((program, action)) = selected else {
                return lower_call(edit, id);
            };
            let mut ctx = RewriteContext::new(id, edit.editor());
            match *action {
                vm::Action::Recipe { entry, slots, .. } => {
                    vm::apply(program, entry, slots, &mut ctx)
                }
                vm::Action::Legal => unreachable!("legal instructions do not rewrite"),
            }
        });
        result?;
        if selected.is_none() && needs_abi(function, id) {
            return Err(Error::codegen("ABI lowering left an unresolved call"));
        }
        if changes.insts.is_empty() {
            return Err(Error::codegen(format!(
                "legalization rule {rule:?} made no instruction edits for {id:?} {opcode:?}"
            )));
        }
        Ok(Some(changes))
    }
}
