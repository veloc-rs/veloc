pub mod info;

pub use info::*;

use crate::error::{Error, Result};
use crate::mir::{GenericOpcode, MachineFunction, MachineInst};
use crate::pipeline::stages::LegalizedMir;
use crate::target::arch::TargetLegalizer;

pub struct Legalizer<'a> {
    target: &'a dyn TargetLegalizer,
}

impl<'a> Legalizer<'a> {
    pub fn new(target: &'a dyn TargetLegalizer) -> Self {
        Self { target }
    }

    pub fn legalize(&self, mfunc: &mut MachineFunction<LegalizedMir>) -> Result<()> {
        let num_blocks = mfunc.blocks.len();
        for i in 0..num_blocks {
            mfunc
                .rewrite_block(i, |cursor| {
                    let inst_id = cursor.current_inst_id();
                    if cursor.current_inst().is_invalid() {
                        cursor.remove_current();
                        return Ok(());
                    }

                    if cursor.current_inst().generic_opcode().is_none() {
                        cursor.keep_current();
                        return Ok(());
                    }

                    let action = {
                        let inst = cursor.current_inst();
                        self.target.legalize_action(inst, cursor.mfunc())?
                    };

                    match action {
                        None => {
                            let (opcode, operands) =
                                self.inst_signature_context(cursor.current_inst(), cursor.mfunc())?;
                            return Err(Error::codegen(alloc::format!(
                                "missing legalization rule for {:?} with signature {:?}",
                                opcode, operands
                            )));
                        }
                        Some(LegalizeAction::Legal) => {
                            cursor.keep_current();
                        }
                        Some(LegalizeAction::Lower) => {
                            let LegalizeResult::Replace(output) = self
                                .target
                                .legalize_instruction(inst_id, cursor.mfunc_mut())?;
                            cursor.remove_current();
                            for new_id in output {
                                cursor.emit_existing_before(new_id);
                            }
                        }
                        Some(LegalizeAction::WidenScalar { to }) => {
                            let (opcode, operands) =
                                self.inst_signature_context(cursor.current_inst(), cursor.mfunc())?;
                            return Err(Error::codegen(alloc::format!(
                                "widen-scalar legalization is not implemented yet for {:?} with signature {:?} (target {:?})",
                                opcode, operands, to
                            )));
                        }
                    }

                    Ok(())
                })
                ?;
        }
        Ok(())
    }

    fn inst_signature_context(
        &self,
        inst: &MachineInst,
        mfunc: &MachineFunction<LegalizedMir>,
    ) -> Result<(GenericOpcode, alloc::string::String)> {
        let opcode = inst
            .generic_opcode()
            .ok_or_else(|| Error::codegen("legalization received a non-generic instruction"))?;
        let operands = format_inst_operands(inst, mfunc)?;
        Ok((opcode, operands))
    }
}
