pub mod info;

pub use info::*;

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use crate::target::arch::TargetLegalizer;
use veloc_lir::{GenericOpcode, MachineFunction};

pub struct Legalizer<'a> {
    target: &'a dyn TargetLegalizer,
}

impl<'a> Legalizer<'a> {
    pub fn new(target: &'a dyn TargetLegalizer) -> Self {
        Self { target }
    }

    pub fn legalize(&self, mfunc: &mut MachineFunction) -> Result<()> {
        // Process expansions in program order, including generic instructions
        // produced by other rules. A single forward scan is not a legalizer.
        const MAX_REWRITES: usize = 1024;
        let mut pending = alloc::vec::Vec::new();
        let mut block = 0;
        while block < mfunc.blocks.len() {
            mfunc.rewrite_block(block, |cursor| {
                pending.clear();
                pending.push(cursor.current_inst_id());
                cursor.detach_current();
                let mut rewrites = 0;
                while let Some(id) = pending.pop() {
                    let inst = &cursor.mfunc().inst(id);
                    if inst.is_invalid() {
                        continue;
                    }
                    if inst.generic_opcode().is_none() {
                        cursor.emit(id);
                        continue;
                    }
                    match self.target.legalize_action(inst, cursor.mfunc())? {
                        None => {
                            let (opcode, operands) = self.inst_signature_context(inst, cursor.mfunc())?;
                            return Err(Error::codegen(alloc::format!(
                                "missing legalization rule for {opcode:?} with signature {operands:?}"
                            )));
                        }
                        Some(LegalizeAction::Legal) => cursor.emit(id),
                        Some(LegalizeAction::Lower) => {
                            if rewrites == MAX_REWRITES {
                                return Err(Error::codegen(alloc::format!(
                                    "legalization did not converge after {MAX_REWRITES} rewrites: {:?}",
                                    inst.opcode()
                                )));
                            }
                            rewrites += 1;
                            let LegalizeResult::Replace(output) =
                                self.target.legalize_instruction(id, cursor.mfunc_mut())?;
                            // Rules may rewrite the same ID in place. Preserve it
                            // in that case and check its new form on the worklist.
                            if !output.contains(&id) {
                                cursor.mfunc_mut().invalidate_inst(id);
                            }
                            pending.extend(output.into_iter().rev());
                        }
                        Some(LegalizeAction::WidenScalar { to }) => {
                            let (opcode, operands) = self.inst_signature_context(inst, cursor.mfunc())?;
                            return Err(Error::codegen(alloc::format!(
                                "widen-scalar legalization is not implemented yet for {opcode:?} with signature {operands:?} (target {to:?})"
                            )));
                        }
                    }
                }
                Ok(())
            })?;
            block += 1;
        }
        Ok(())
    }

    fn inst_signature_context(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> Result<(GenericOpcode, alloc::string::String)> {
        let opcode = inst
            .generic_opcode()
            .ok_or_else(|| Error::codegen("legalization received a non-generic instruction"))?;
        let operands = format_inst_operands(inst, mfunc)?;
        Ok((opcode, operands))
    }
}
