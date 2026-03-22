//! x86_64 Machine Code Emitter
//!
//! x86_64 后端只保留发射时机和 fixup 收尾逻辑。
//! 具体编码字节序列由 ISLE DSL 生成的 `TargetInst::emit()` 负责。

use crate::mir::{MachineBlock, MachineFunction, MachineInst, MachineOpcode};
use crate::target::arch::TargetEmitter;

/// x86_64 机器码发射器实现
pub struct X86_64CodeEmitter;

impl X86_64CodeEmitter {
    pub fn new() -> Self {
        Self
    }
}

impl TargetEmitter for X86_64CodeEmitter {
    fn begin_block(
        &self,
        emitter: &mut crate::Emitter,
        block: &MachineBlock,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        emitter.mark_block(block.id);
        Ok(())
    }

    fn emit_instruction(
        &self,
        emitter: &mut crate::Emitter,
        inst: &MachineInst,
        mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        match &inst.opcode {
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
                let target = crate::target::x86_64::isle::TargetInst::from_u32(*target_inst_code);
                target.emit::<Self>(emitter, inst, mfunc).unwrap_or_else(|err| {
                    panic!("{err} | inst={:?}", inst);
                });
                Ok(())
            }
        }
    }

    fn finish_function(
        &self,
        emitter: &mut crate::Emitter,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        emitter.apply_fixups()
    }
}
