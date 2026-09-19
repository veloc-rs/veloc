//! x86_64 Machine Code Emitter
//!
//! Adapts allocated LIR to the standalone encoder. Frame and symbol knowledge
//! stays here; architecture encoding algorithms live in veloc-encoder.

use crate::target::TargetEmitter;
use veloc_lir::{MachineFunction, MachineOpcode};

/// x86_64 机器码发射器实现
pub struct X86_64CodeEmitter {
    features: super::inst::FeatureSet,
}

impl X86_64CodeEmitter {
    pub fn new(features: super::inst::FeatureSet) -> Self {
        Self { features }
    }
}

impl TargetEmitter for X86_64CodeEmitter {
    fn begin_block(
        &self,
        emitter: &mut crate::Emitter,
        block: veloc_lir::BlockId,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        emitter.mark_block(block);
        Ok(())
    }

    fn emit_instruction(
        &self,
        emitter: &mut crate::Emitter,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        match &inst.opcode() {
            MachineOpcode::Invalid => {
                panic!("invalid opcode cannot be emitted: {:?}", inst);
            }
            MachineOpcode::Generic(_) => {
                panic!(
                    "generic opcode should be translated to target opcode before emission: {:?}",
                    inst
                );
            }
            MachineOpcode::Target(target_inst_code) => {
                let target = crate::target::x86_64::inst::TargetInst::from_u32(*target_inst_code);
                if target.is_pseudo() || !target.has_encoding() {
                    return Err(crate::Error::codegen(alloc::format!(
                        "{target:?} has no final encoding"
                    )));
                }
                if !self.features.contains_all(target.required_features()) {
                    return Err(crate::Error::codegen(alloc::format!(
                        "{target:?} requires unavailable target features"
                    )));
                }
                if let Some(access) = inst.memory() {
                    let shape = super::inst::target_inst_metadata(target).memory;
                    if shape != Some((access.kind, access.bytes))
                        || !access.alignment.is_power_of_two()
                    {
                        return Err(crate::Error::codegen(
                            "emitted instruction violates its memory access contract",
                        ));
                    }
                }
                target.emit(emitter, inst, mfunc).map_err(|error| {
                    crate::Error::codegen(alloc::format!("{target:?}: {error}; {inst:?}"))
                })
            }
        }
    }
}

pub(crate) fn register(reg: veloc_lir::Reg) -> crate::Result<veloc_encoder::x86_64::Reg> {
    super::inst::register_encoding(reg)
        .and_then(veloc_encoder::x86_64::Reg::new)
        .ok_or_else(|| crate::Error::codegen("invalid physical register for x86 encoding"))
}

pub(crate) fn stack_address(
    frame: &veloc_lir::StackFrame,
    slot: veloc_lir::StackSlot,
) -> crate::Result<veloc_encoder::x86_64::Address> {
    use veloc_encoder::x86_64::{Address, Memory};
    let slot = frame.address(slot);
    Ok(Address::BaseIndex(Memory {
        base: Some(register(slot.base)?),
        index: None,
        displacement: i64::from(slot.offset),
    }))
}

// Generated traits check every declared host signature, including methods that
// no current instruction happens to call.
pub(crate) mod host {
    use veloc_encoder::x86_64::{Branch, Form, Immediate, Legacy};
    include!(concat!(env!("OUT_DIR"), "/encoding_host.rs"));
}

/// Codegen owns symbolic targets; the standalone encoder never sees them.
pub(crate) enum Emission {
    Legacy(
        veloc_encoder::x86_64::Legacy,
        veloc_encoder::x86_64::Form,
        veloc_encoder::x86_64::Immediate,
    ),
    Branch(veloc_lir::BlockId, veloc_encoder::x86_64::Branch),
    Relative(
        veloc_lir::SymbolId,
        veloc_encoder::x86_64::Legacy,
        veloc_encoder::x86_64::Form,
        i64,
    ),
}

impl host::Emission for Emission {
    fn legacy(
        descriptor: veloc_encoder::x86_64::Legacy,
        form: veloc_encoder::x86_64::Form,
        immediate: veloc_encoder::x86_64::Immediate,
    ) -> Self {
        Self::Legacy(descriptor, form, immediate)
    }
    fn branch(target: veloc_lir::BlockId, form: veloc_encoder::x86_64::Branch) -> Self {
        Self::Branch(target, form)
    }
    fn relative(
        target: veloc_lir::SymbolId,
        descriptor: veloc_encoder::x86_64::Legacy,
        form: veloc_encoder::x86_64::Form,
        addend: i64,
    ) -> Self {
        Self::Relative(target, descriptor, form, addend)
    }
}

pub(crate) fn encode_instruction(
    emitter: &mut crate::Emitter,
    emission: Emission,
) -> crate::Result<()> {
    use veloc_encoder::x86_64 as x86;
    let error = |e| crate::Error::codegen(alloc::format!("x86 encoding: {e}"));
    match emission {
        Emission::Legacy(descriptor, form, immediate) => {
            let encoded = x86::encode(descriptor, form, immediate).map_err(error)?;
            emitter.instruction(&encoded, None)
        }
        Emission::Relative(target, descriptor, form, addend) => {
            let encoded =
                x86::encode(descriptor, form, x86::Immediate::Relative(addend)).map_err(error)?;
            emitter.instruction(&encoded, Some(crate::emitter::Target::Symbol(target)))
        }
        Emission::Branch(target, branch) => {
            let short = x86::encode_branch(branch, true).map_err(error)?;
            let long = x86::encode_branch(branch, false).map_err(error)?;
            emitter.branch(target, &short, &long)
        }
    }
}
