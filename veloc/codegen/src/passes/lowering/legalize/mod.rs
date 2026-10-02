mod bytecode;
pub mod info;
pub mod vm;

pub use info::*;

use crate::error::{Error, Result};
use crate::target::TargetMachine;
use std::collections::VecDeque;
use veloc_lir::function::EditChanges;
use veloc_lir::{InstId, MachineFunction, SymbolTable};

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
            check_call_abi(function, id)?;
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
    /// Explicit libcall actions invoke ABI lowering to create a fully lowered
    /// call. Ordinary calls must already have passed through ABI lowering.
    pub fn legalize(
        &self,
        function: &mut MachineFunction,
        target: &dyn TargetMachine,
        symbols: &mut SymbolTable,
    ) -> Result<bool> {
        let mut modified = false;
        let mut pending: VecDeque<_> = function
            .blocks()
            .flat_map(|block| function.block_insts(block))
            .collect();
        let mut queued: hashbrown::HashSet<_> = pending.iter().copied().collect();
        while let Some(id) = pending.pop_front() {
            queued.remove(&id);
            let Some(changes) = self.step(function, id, target, symbols)? else {
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

fn check_call_abi(function: &MachineFunction, id: InstId) -> Result<()> {
    if function
        .try_call_info(id)
        .is_some_and(|info| info.frame.is_none())
    {
        return Err(Error::codegen(format!("unlowered ABI call {id:?}")));
    }
    Ok(())
}

impl Legalizer<'_> {
    fn step(
        &self,
        function: &mut MachineFunction,
        id: InstId,
        target: &dyn TargetMachine,
        symbols: &mut SymbolTable,
    ) -> Result<Option<EditChanges>> {
        if function.inst_block(id).is_none() {
            return Ok(None);
        }
        let inst = function.inst(id);
        if !inst.is_generic() || inst.is_call_frame() {
            return Ok(None);
        }
        let opcode = inst.opcode();
        let (program, action) = vm::select(self.target, function, id)?
            .ok_or_else(|| Error::codegen(format!("missing legalization rule for {opcode:?}")))?;
        let rule = match action {
            vm::Action::Legal => return Ok(None),
            vm::Action::Recipe { name, .. } | vm::Action::Libcall { name, .. } => name,
        };
        if !inst.constraints().is_empty() {
            return Err(Error::codegen(
                "constrained ABI boundary requires value conversion before ABI lowering",
            ));
        }
        let (result, changes) = function.editor().track(|edit| match *action {
            vm::Action::Recipe { entry, slots, .. } => {
                let mut ctx = RewriteContext::new(id, edit.editor());
                vm::apply(program, entry, slots, &mut ctx)
            }
            vm::Action::Libcall { symbol, .. } => {
                super::abi::emit_libcall(target, symbols, edit, id, symbol)
            }
            vm::Action::Legal => unreachable!("legal instructions do not rewrite"),
        });
        result?;
        if changes.insts.is_empty() {
            return Err(Error::codegen(format!(
                "legalization rule {rule:?} made no instruction edits for {id:?} {opcode:?}"
            )));
        }
        Ok(Some(changes))
    }
}
